"""
MediaPipe Pose Landmark Validator
Runs local (offline) full-body pose detection using MediaPipe Tasks API.
Returns 33 3D body landmarks for anthropometric analysis.

Detection is best-effort — head/shoulder images will return partial results.
Full-body frontal capture yields the most complete measurement set.
"""

import numpy as np
from pathlib import Path
from PIL import Image

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

PIPELINE_VERSION = "1.0.0"
DETECTION_SCALES = [1024, 640, 1920]

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH  = MODELS_DIR / "pose_landmarker.task"

# Canonical names for all 33 MediaPipe PoseLandmarker landmarks
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _get_detector(detection_confidence: float = 0.3):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"PoseLandmarker model not found at {MODEL_PATH}. "
            "Run: bash scripts/setup_models.sh"
        )
    base_options = mp_tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=mp_tasks.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1,
        min_pose_detection_confidence=detection_confidence,
        min_pose_presence_confidence=detection_confidence,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def _scale_image(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def validate_pose(image_path: str | Path) -> dict | None:
    """
    Run MediaPipe PoseLandmarker on a pre-processed (EXIF-stripped) image.

    Args:
        image_path: Path to the clean image

    Returns:
        body_pose dict, or None if no pose detected
    """
    import mediapipe as mp

    image_path = Path(image_path)
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")

    detector = _get_detector()
    results  = None

    for max_side in DETECTION_SCALES:
        scaled   = _scale_image(rgb, max_side)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.array(scaled),
        )
        results = detector.detect(mp_image)
        if results.pose_landmarks:
            break

    if not results or not results.pose_landmarks:
        return None

    pose        = results.pose_landmarks[0]
    landmarks   = []
    visibilities = []

    for i, lm in enumerate(pose):
        vis = getattr(lm, "visibility", 1.0) or 1.0
        landmarks.append({
            "index":      i,
            "name":       LANDMARK_NAMES[i],
            "x":          round(lm.x, 6),
            "y":          round(lm.y, 6),
            "z":          round(lm.z, 6),
            "visibility": round(float(vis), 4),
        })
        visibilities.append(float(vis))

    # Identify which body regions are visible (visibility > 0.5)
    visible = {lm["name"] for lm in landmarks if lm["visibility"] > 0.5}
    coverage = _coverage_flags(visible)

    return {
        "landmarks":      landmarks,
        "confidence":     round(float(np.mean(visibilities)), 6),
        "validator":      "pose_landmarker_full",
        "landmark_count": len(landmarks),
        "coverage":       coverage,
    }


def _coverage_flags(visible_names: set) -> dict:
    """Summarise which body regions have sufficient landmark visibility."""
    def all_visible(*names):
        return all(n in visible_names for n in names)

    return {
        "head":         all_visible("nose", "left_ear", "right_ear"),
        "upper_body":   all_visible("left_shoulder", "right_shoulder"),
        "arms":         all_visible("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
        "torso":        all_visible("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
        "lower_body":   all_visible("left_hip", "right_hip", "left_knee", "right_knee"),
        "full_legs":    all_visible("left_knee", "right_knee", "left_ankle", "right_ankle"),
        "feet":         all_visible("left_foot_index", "right_foot_index"),
    }


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python -m agents.pose_validator <image_path>")
        sys.exit(1)

    result = validate_pose(sys.argv[1])
    if result:
        print(f"Detected {result['landmark_count']} landmarks (confidence={result['confidence']})")
        print(f"Coverage: {result['coverage']}")
        print(json.dumps(result['landmarks'][:5], indent=2), "...")
    else:
        print("No pose detected.")
