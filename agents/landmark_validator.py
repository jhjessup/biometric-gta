"""
MediaPipe Landmark Validator
Runs local (offline) face landmark detection using MediaPipe Tasks API.
All inference is CPU-only — no network calls.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_CONFIDENCE = 0.85
MIN_AVG_VISIBILITY = 0.90

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def _get_detector():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {MODEL_PATH}. "
            "Run: bash scripts/setup_models.sh"
        )
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def validate_landmarks(image_path: str | Path, image_hash: str) -> dict:
    """
    Run MediaPipe FaceLandmarker on a pre-processed (EXIF-stripped) image.

    Args:
        image_path: Path to the clean image
        image_hash: SHA-256 of the image (from exif_stripper)

    Returns:
        GTA artifact dict conforming to anatomy/landmark_schema.json
    """
    import mediapipe as mp

    image_path = Path(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
        rgb = img.convert("RGB")

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.array(rgb),
    )

    detector = _get_detector()
    results = detector.detect(mp_image)

    if not results.face_landmarks:
        raise ValueError("No face detected in image.")

    face = results.face_landmarks[0]
    quality_flags = []

    landmarks = []
    visibilities = []
    x_coords, y_coords = [], []

    for i, lm in enumerate(face):
        vis = getattr(lm, "visibility", 1.0) or 1.0
        landmarks.append({
            "index": i,
            "x": round(lm.x, 6),
            "y": round(lm.y, 6),
            "z": round(lm.z, 6),
            "visibility": round(vis, 4),
        })
        visibilities.append(vis)
        x_coords.append(lm.x)
        y_coords.append(lm.y)

    confidence = float(np.mean(visibilities))

    if confidence < MIN_CONFIDENCE:
        quality_flags.append("low_confidence")
    if np.mean(visibilities) < MIN_AVG_VISIBILITY:
        quality_flags.append("partial_occlusion")

    bounding_box = {
        "x_min": round(min(x_coords), 6),
        "y_min": round(min(y_coords), 6),
        "x_max": round(max(x_coords), 6),
        "y_max": round(max(y_coords), 6),
    }

    artifact = {
        "artifact_id": str(uuid.uuid4()),
        "schema_version": SCHEMA_VERSION,
        "source": {
            "image_hash": image_hash,
            "capture_timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": {"width": width, "height": height},
        },
        "landmarks": {
            "face_mesh": landmarks,
            "confidence": round(confidence, 6),
            "validator": "face_landmarker_v2",
            "bounding_box": bounding_box,
        },
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
            "quality_flags": quality_flags,
            "approved": False,
        },
    }

    return artifact


def save_artifact(artifact: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python landmark_validator.py <image_path> <image_hash> <output_artifact.json>")
        sys.exit(1)

    img_path, img_hash, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    artifact = validate_landmarks(img_path, img_hash)
    save_artifact(artifact, out_path)
    flags = artifact["metadata"]["quality_flags"]
    print(f"Artifact: {artifact['artifact_id']}")
    print(f"Confidence: {artifact['landmarks']['confidence']}")
    print(f"Landmarks: {len(artifact['landmarks']['face_mesh'])}")
    print(f"Quality flags: {flags or 'none'}")
    print(f"Saved to: {out_path}")
