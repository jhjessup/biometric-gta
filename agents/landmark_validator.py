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

# Detection scales tried in order — MediaPipe works best at lower resolutions
# for high-megapixel source images.
DETECTION_SCALES = [1024, 640, 1920]

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def _get_detector(detection_confidence: float = 0.3):
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
        min_face_detection_confidence=detection_confidence,
        min_face_presence_confidence=detection_confidence,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _scale_image(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale image so its longest side = max_side, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _get_face_crops(img: Image.Image) -> tuple[list[tuple[Image.Image, tuple]], bool]:
    """
    Use OpenCV Haar cascade to locate all faces and return padded crops sorted by area (largest first).

    Returns:
        ([(crop_img, (x1,y1,w,h)), ...], multiple_faces_flag)
    """
    import cv2

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    orig_w, orig_h = img.size
    gray_full = np.array(img.convert("L"))

    all_faces = []
    for detect_scale in [0.25, 0.15, 0.5, 0.35]:
        small = cv2.resize(gray_full, (int(orig_w * detect_scale), int(orig_h * detect_scale)))
        faces = cascade.detectMultiScale(small, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
        if len(faces) > 0:
            all_faces = [[int(v / detect_scale) for v in f] for f in faces]
            break

    if not all_faces:
        return [], False

    # Sort largest area first
    all_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    multiple = len(all_faces) > 1

    crops = []
    for x, y, w, h in all_faces:
        pad_x = int(w * 0.7)
        pad_y = int(h * 0.7)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(orig_w, x + w + pad_x)
        y2 = min(orig_h, y + h + pad_y)
        crops.append((img.crop((x1, y1, x2, y2)), (x1, y1, x2 - x1, y2 - y1)))

    return crops, multiple


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

    # Try detection at multiple scales — high-res images often need downscaling
    results = None
    used_scale = None
    face_crop_box = None
    multiple_faces_flag = False
    detector = _get_detector()

    for max_side in DETECTION_SCALES:
        scaled = _scale_image(rgb, max_side)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(scaled),
        )
        results = detector.detect(mp_image)
        if results.face_landmarks:
            used_scale = max_side
            break

    # Fallback: Haar crop + MediaPipe — try each detected face until one succeeds
    if not results or not results.face_landmarks:
        crops, multiple_faces_flag = _get_face_crops(rgb)
        for crop, crop_box in crops:
            for max_side in DETECTION_SCALES:
                scaled_crop = _scale_image(crop, max_side)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.array(scaled_crop),
                )
                results = detector.detect(mp_image)
                if results.face_landmarks:
                    used_scale = max_side
                    face_crop_box = crop_box
                    break
            if face_crop_box is not None:
                break

    if not results or not results.face_landmarks:
        raise ValueError("No face detected in image at any resolution scale or via face crop.")

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
    if face_crop_box is not None:
        quality_flags.append("face_crop_used")
    if multiple_faces_flag:
        quality_flags.append("multiple_faces_detected")

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
            "detection_scale": used_scale,
            "face_crop_box": face_crop_box,
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
