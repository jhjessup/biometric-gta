import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agents.exif_stripper import strip_exif, strip_exif_to_file, verify_no_exif
from agents.landmark_validator import validate_landmarks, save_artifact


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

class MockLandmark:
    def __init__(self, x: float, y: float, z: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockDetectorResult:
    def __init__(self, landmarks):
        # face_landmarks[0] is the list of per-landmark objects
        self.face_landmarks = [landmarks] if landmarks else []


def _make_landmarks(n: int = 468, visibility: float = 0.99) -> list[MockLandmark]:
    return [MockLandmark(0.1 + i * 0.001, 0.1 + i * 0.001, 0.0, visibility) for i in range(n)]


@pytest.fixture
def plain_image(tmp_path: Path) -> Path:
    """10×10 RGB image with no metadata."""
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (10, 10), (100, 100, 100)).save(p, format="JPEG")
    return p


@pytest.fixture
def dirty_image(tmp_path: Path) -> Path:
    """10×10 JPEG saved with fake EXIF and ICC bytes in info."""
    p = tmp_path / "dirty.jpg"
    img = Image.new("RGB", (10, 10), (100, 100, 100))
    fake_exif = b"Exif\x00\x00" + b"\xDE\xAD" * 20
    fake_icc = b"fake_icc_profile_data"
    img.save(p, format="JPEG", exif=fake_exif, icc_profile=fake_icc)
    return p


# ---------------------------------------------------------------------------
# exif_stripper tests
# ---------------------------------------------------------------------------

class TestExifStripper:
    def test_strip_exif_returns_bytes_and_sha256(self, plain_image):
        image_bytes, sha256 = strip_exif(plain_image)
        assert isinstance(image_bytes, bytes) and len(image_bytes) > 0
        assert sha256 == hashlib.sha256(image_bytes).hexdigest()
        assert len(sha256) == 64

    def test_strip_exif_deterministic(self, plain_image):
        _, h1 = strip_exif(plain_image)
        _, h2 = strip_exif(plain_image)
        assert h1 == h2

    def test_strip_exif_to_file_writes_and_returns_hash(self, plain_image, tmp_path):
        out = tmp_path / "clean.png"
        sha = strip_exif_to_file(plain_image, out)
        assert out.exists()
        assert hashlib.sha256(out.read_bytes()).hexdigest() == sha

    def test_strip_exif_to_file_creates_parent_dirs(self, plain_image, tmp_path):
        out = tmp_path / "nested" / "dir" / "clean.png"
        strip_exif_to_file(plain_image, out)
        assert out.exists()

    def test_verify_no_exif_clean_image(self, plain_image, tmp_path):
        out = tmp_path / "clean.png"
        strip_exif_to_file(plain_image, out)
        assert verify_no_exif(out) is True

    def test_verify_no_exif_dirty_image(self, dirty_image):
        # After saving with exif/icc bytes, PIL should expose them on re-open
        with Image.open(dirty_image) as img:
            has_exif = len(img.info.get("exif", b"")) > 0
            has_icc = len(img.info.get("icc_profile", b"")) > 0
        if has_exif or has_icc:
            assert verify_no_exif(dirty_image) is False


# ---------------------------------------------------------------------------
# landmark_validator tests
# ---------------------------------------------------------------------------

class TestLandmarkValidator:
    @pytest.fixture
    def test_image(self, tmp_path):
        p = tmp_path / "face.jpg"
        Image.new("RGB", (640, 480), (40, 40, 40)).save(p, format="JPEG")
        return p

    @patch("agents.landmark_validator._get_detector")
    def test_artifact_required_keys(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks()))
        )
        art = validate_landmarks(test_image, "abc123")
        assert {"artifact_id", "schema_version", "source", "landmarks", "metadata"}.issubset(art)

    @patch("agents.landmark_validator._get_detector")
    def test_source_structure(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks()))
        )
        art = validate_landmarks(test_image, "abc123")
        src = art["source"]
        assert src["image_hash"] == "abc123"
        assert "capture_timestamp" in src
        assert src["resolution"]["width"] == 640
        assert src["resolution"]["height"] == 480

    @patch("agents.landmark_validator._get_detector")
    def test_landmarks_structure(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks(468)))
        )
        art = validate_landmarks(test_image, "abc123")
        lm = art["landmarks"]
        assert "face_mesh" in lm and "confidence" in lm and "validator" in lm
        assert len(lm["face_mesh"]) == 468
        assert 0.0 <= lm["confidence"] <= 1.0

    @patch("agents.landmark_validator._get_detector")
    def test_metadata_structure(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks()))
        )
        art = validate_landmarks(test_image, "abc123")
        meta = art["metadata"]
        assert "quality_flags" in meta
        assert isinstance(meta["quality_flags"], list)
        assert "approved" in meta
        assert meta["approved"] is False

    @patch("agents.landmark_validator._get_detector")
    def test_no_quality_flags_high_confidence(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks(visibility=0.99)))
        )
        art = validate_landmarks(test_image, "abc123")
        flags = art["metadata"]["quality_flags"]
        assert "low_confidence" not in flags
        assert "face_crop_used" not in flags

    @patch("agents.landmark_validator._get_detector")
    def test_low_confidence_flag(self, mock_get_detector, test_image):
        mock_get_detector.return_value = MagicMock(
            detect=MagicMock(return_value=MockDetectorResult(_make_landmarks(visibility=0.50)))
        )
        art = validate_landmarks(test_image, "abc123")
        flags = art["metadata"]["quality_flags"]
        assert "low_confidence" in flags

    def test_save_artifact_roundtrip(self, tmp_path):
        artifact = {"artifact_id": "test-001", "data": [1, 2, 3]}
        out = tmp_path / "artifact.json"
        save_artifact(artifact, out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == artifact

    def test_save_artifact_creates_parent_dirs(self, tmp_path):
        artifact = {"x": 1}
        out = tmp_path / "nested" / "artifacts" / "a.json"
        save_artifact(artifact, out)
        assert out.exists()
