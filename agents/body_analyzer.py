"""
Body Geometry Analyzer
Derives anthropometric measurements from stored MediaPipe PoseLandmarker coordinates.
Operates entirely offline on existing artifact JSON — no image required.

All linear measurements are first normalized by shoulder width (SW), then converted
to estimated centimetres using IOD calibration (population mean IOD = 63 mm).

IOD is sourced from the artifact's geometry block (iod_raw in normalized image coords).
If geometry is absent, pose-only normalization is used and cm estimates are omitted.

MediaPipe PoseLandmarker 33-point landmark indices used here match the
canonical BlazePose GHUM topology.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ANALYZER_VERSION = "1.0.0"

# Population mean inter-ocular distance used for pixel→mm calibration
IOD_CALIBRATION_MM = 63.0

# ---------------------------------------------------------------------------
# Canonical MediaPipe PoseLandmarker landmark indices
# ---------------------------------------------------------------------------
PL = {
    "nose":               0,
    "left_eye_inner":     1,
    "left_eye":           2,
    "left_eye_outer":     3,
    "right_eye_inner":    4,
    "right_eye":          5,
    "right_eye_outer":    6,
    "left_ear":           7,
    "right_ear":          8,
    "mouth_left":         9,
    "mouth_right":        10,
    "left_shoulder":      11,
    "right_shoulder":     12,
    "left_elbow":         13,
    "right_elbow":        14,
    "left_wrist":         15,
    "right_wrist":        16,
    "left_pinky":         17,
    "right_pinky":        18,
    "left_index":         19,
    "right_index":        20,
    "left_thumb":         21,
    "right_thumb":        22,
    "left_hip":           23,
    "right_hip":          24,
    "left_knee":          25,
    "right_knee":         26,
    "left_ankle":         27,
    "right_ankle":        28,
    "left_heel":          29,
    "right_heel":         30,
    "left_foot_index":    31,
    "right_foot_index":   32,
}

# Minimum visibility to include a landmark in calculations
VIS_THRESHOLD = 0.4


def _dist2d(p1: dict, p2: dict) -> float:
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def _dist3d(p1: dict, p2: dict) -> float:
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2 +
        (p1["y"] - p2["y"]) ** 2 +
        (p1.get("z", 0) - p2.get("z", 0)) ** 2
    )


def _midpoint(p1: dict, p2: dict) -> dict:
    return {
        "x": (p1["x"] + p2["x"]) / 2,
        "y": (p1["y"] + p2["y"]) / 2,
        "z": (p1.get("z", 0) + p2.get("z", 0)) / 2,
    }


def analyze_body(artifact: dict) -> dict:
    """
    Compute anthropometric body measurements from a GTA artifact dict.

    Requires artifact['body_pose']['landmarks'] (from pose_validator).
    Optionally uses artifact['geometry']['iod_raw'] for cm calibration.

    Args:
        artifact: Parsed artifact JSON containing body_pose.landmarks

    Returns:
        body_geometry dict suitable for artifact['body_geometry']
    """
    body_pose = artifact.get("body_pose")
    if not body_pose or not body_pose.get("landmarks"):
        raise ValueError("artifact.body_pose.landmarks is required for body geometry analysis")

    # Build landmark lookup by index
    raw = {lm["index"]: lm for lm in body_pose["landmarks"]}

    def pt(name: str) -> dict | None:
        """Return landmark if visible, else None."""
        idx = PL[name]
        lm = raw.get(idx)
        if lm is None:
            return None
        if lm.get("visibility", 1.0) < VIS_THRESHOLD:
            return None
        return lm

    def safe_dist2d(n1: str, n2: str) -> float | None:
        p1, p2 = pt(n1), pt(n2)
        return _dist2d(p1, p2) if p1 and p2 else None

    def safe_dist3d(n1: str, n2: str) -> float | None:
        p1, p2 = pt(n1), pt(n2)
        return _dist3d(p1, p2) if p1 and p2 else None

    # ------------------------------------------------------------------
    # Calibration: derive px-per-mm scale from IOD if available
    # ------------------------------------------------------------------
    cm_per_unit: float | None = None
    iod_raw = artifact.get("geometry", {}).get("iod_raw")
    if iod_raw and iod_raw > 0:
        # iod_raw is in normalized image coords; IOD_CALIBRATION_MM is in mm
        # => 1 normalized unit = IOD_CALIBRATION_MM / iod_raw mm
        mm_per_unit = IOD_CALIBRATION_MM / iod_raw
        cm_per_unit = mm_per_unit / 10.0

    def to_cm(d: float | None) -> float | None:
        if d is None or cm_per_unit is None:
            return None
        return round(d * cm_per_unit, 1)

    def ratio(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or b == 0:
            return None
        return round(a / b, 4)

    m = {}

    # ------------------------------------------------------------------
    # 1. Shoulder measurements
    # ------------------------------------------------------------------
    shoulder_w = safe_dist2d("left_shoulder", "right_shoulder")
    m["shoulder_width_norm"]  = round(shoulder_w, 6) if shoulder_w else None
    m["shoulder_width_cm"]    = to_cm(shoulder_w)
    m["shoulder_width_note"]  = "Biacromial width estimated from anterior shoulder landmarks"

    # Shoulder midpoint (used as torso reference)
    ls, rs = pt("left_shoulder"), pt("right_shoulder")
    shoulder_mid = _midpoint(ls, rs) if ls and rs else None

    # ------------------------------------------------------------------
    # 2. Hip measurements
    # ------------------------------------------------------------------
    hip_w = safe_dist2d("left_hip", "right_hip")
    m["hip_width_norm"]  = round(hip_w, 6) if hip_w else None
    m["hip_width_cm"]    = to_cm(hip_w)
    m["hip_width_note"]  = "Bi-iliac width (anterior) from hip landmarks; lateral circumference not available from single frontal view"

    lh, rh = pt("left_hip"), pt("right_hip")
    hip_mid = _midpoint(lh, rh) if lh and rh else None

    # ------------------------------------------------------------------
    # 3. Waist estimate
    # Midpoint between shoulder_mid and hip_mid in image space
    # ------------------------------------------------------------------
    if shoulder_mid and hip_mid:
        waist_mid_y = (shoulder_mid["y"] + hip_mid["y"]) / 2
        # Estimate waist width as 85% of shoulder width (population average proxy)
        # This is a rough anthropometric estimate; multi-angle capture improves this
        waist_est = shoulder_w * 0.85 if shoulder_w else None
        m["waist_width_cm"]    = to_cm(waist_est)
        m["waist_width_note"]  = (
            "Estimated as 85% of shoulder width (population mean proxy). "
            "Actual waist requires multi-angle capture or depth sensor."
        )
    else:
        m["waist_width_cm"]   = None
        m["waist_width_note"] = "Insufficient landmark visibility"

    # ------------------------------------------------------------------
    # 4. Chest / bust estimate
    # Width at armpit level ≈ shoulder_width * 0.92 (average population ratio)
    # ------------------------------------------------------------------
    chest_est = shoulder_w * 0.92 if shoulder_w else None
    m["chest_width_cm"]   = to_cm(chest_est)
    m["chest_width_note"] = (
        "Estimated as 92% of shoulder width (population mean proxy for chest breadth). "
        "Bust circumference not derivable from 2D pose landmarks."
    )

    # ------------------------------------------------------------------
    # 5. Torso measurements
    # ------------------------------------------------------------------
    torso_len = safe_dist2d("left_shoulder", "left_hip") if pt("left_shoulder") and pt("left_hip") else None
    if torso_len is None:
        torso_len = safe_dist2d("right_shoulder", "right_hip")
    torso_via_mid = (
        _dist2d(shoulder_mid, hip_mid) if shoulder_mid and hip_mid else None
    )

    m["torso_length_norm"] = round(torso_via_mid, 6) if torso_via_mid else None
    m["torso_length_cm"]   = to_cm(torso_via_mid)

    # ------------------------------------------------------------------
    # 6. Arm measurements
    # ------------------------------------------------------------------
    upper_arm_left  = safe_dist3d("left_shoulder",  "left_elbow")
    lower_arm_left  = safe_dist3d("left_elbow",     "left_wrist")
    upper_arm_right = safe_dist3d("right_shoulder", "right_elbow")
    lower_arm_right = safe_dist3d("right_elbow",    "right_wrist")

    arm_left  = (upper_arm_left  + lower_arm_left)  if (upper_arm_left  and lower_arm_left)  else None
    arm_right = (upper_arm_right + lower_arm_right) if (upper_arm_right and lower_arm_right) else None

    m["upper_arm_left_cm"]  = to_cm(upper_arm_left)
    m["upper_arm_right_cm"] = to_cm(upper_arm_right)
    m["lower_arm_left_cm"]  = to_cm(lower_arm_left)
    m["lower_arm_right_cm"] = to_cm(lower_arm_right)
    m["arm_left_cm"]        = to_cm(arm_left)
    m["arm_right_cm"]       = to_cm(arm_right)
    m["arm_symmetry"]       = ratio(arm_left, arm_right)

    # ------------------------------------------------------------------
    # 7. Leg measurements
    # ------------------------------------------------------------------
    upper_leg_left  = safe_dist3d("left_hip",   "left_knee")
    lower_leg_left  = safe_dist3d("left_knee",  "left_ankle")
    upper_leg_right = safe_dist3d("right_hip",  "right_knee")
    lower_leg_right = safe_dist3d("right_knee", "right_ankle")

    leg_left  = (upper_leg_left  + lower_leg_left)  if (upper_leg_left  and lower_leg_left)  else None
    leg_right = (upper_leg_right + lower_leg_right) if (upper_leg_right and lower_leg_right) else None

    m["upper_leg_left_cm"]  = to_cm(upper_leg_left)
    m["upper_leg_right_cm"] = to_cm(upper_leg_right)
    m["lower_leg_left_cm"]  = to_cm(lower_leg_left)
    m["lower_leg_right_cm"] = to_cm(lower_leg_right)
    m["leg_left_cm"]        = to_cm(leg_left)
    m["leg_right_cm"]       = to_cm(leg_right)
    m["leg_symmetry"]       = ratio(leg_left, leg_right)

    # ------------------------------------------------------------------
    # 8. Height estimate
    # Strategy: nose-to-ankle (visible body span) + estimated head height
    # Head height above nose ≈ facial_third_upper * face_height (from geometry)
    # Remaining foot-to-ground ≈ heel-to-foot_index distance
    # ------------------------------------------------------------------
    nose_lm = pt("nose")

    # Best ankle estimate (average of both if available)
    ankle_pts = [p for p in [pt("left_ankle"), pt("right_ankle")] if p]
    ankle_y = sum(p["y"] for p in ankle_pts) / len(ankle_pts) if ankle_pts else None

    # Heel estimate (lower of heel/ankle)
    heel_pts = [p for p in [pt("left_heel"), pt("right_heel")] if p]
    foot_pts = [p for p in [pt("left_foot_index"), pt("right_foot_index")] if p]
    ground_candidates = heel_pts + foot_pts
    ground_y = max(p["y"] for p in ground_candidates) if ground_candidates else ankle_y

    nose_to_ground: float | None = None
    if nose_lm and ground_y:
        nose_to_ground = abs(ground_y - nose_lm["y"])

    # Head height above nose (from face geometry if available)
    head_above_nose: float | None = None
    geom = artifact.get("geometry", {})
    face_iod = geom.get("iod_raw")
    face_m   = geom.get("measurements", {})
    if face_iod and face_m.get("facial_third_upper") and face_m.get("face_height_norm"):
        # face_height_norm is face height in IOD units; convert back to normalized image coords
        face_h_norm = face_m["face_height_norm"] * face_iod  # in normalized image coords
        head_above_nose = face_h_norm * face_m["facial_third_upper"]  # forehead portion

    full_height_norm: float | None = None
    if nose_to_ground is not None and head_above_nose is not None:
        full_height_norm = nose_to_ground + head_above_nose
    elif nose_to_ground is not None:
        # Estimate head = ~12.5% of total height (population mean)
        full_height_norm = nose_to_ground / 0.875

    m["height_cm"] = to_cm(full_height_norm)
    m["height_note"] = (
        "Estimated from pose landmark span (nose→ground) + head height from facial geometry. "
        "Accuracy ±5–10% from single frontal image; calibrate against known reference for precision."
    )

    # ------------------------------------------------------------------
    # 9. Body proportion ratios
    # ------------------------------------------------------------------
    m["shoulder_to_hip_ratio"]  = ratio(shoulder_w, hip_w)
    m["torso_to_leg_ratio"]     = ratio(
        torso_via_mid,
        ((leg_left or 0) + (leg_right or 0)) / max(sum(1 for x in [leg_left, leg_right] if x), 1)
        if (leg_left or leg_right) else None
    )
    m["arm_to_leg_ratio"]       = ratio(
        ((arm_left or 0) + (arm_right or 0)) / max(sum(1 for x in [arm_left, arm_right] if x), 1)
        if (arm_left or arm_right) else None,
        ((leg_left or 0) + (leg_right or 0)) / max(sum(1 for x in [leg_left, leg_right] if x), 1)
        if (leg_left or leg_right) else None,
    )
    m["upper_to_lower_arm_ratio"] = ratio(
        ((upper_arm_left or 0) + (upper_arm_right or 0)) / max(sum(1 for x in [upper_arm_left, upper_arm_right] if x), 1)
        if (upper_arm_left or upper_arm_right) else None,
        ((lower_arm_left or 0) + (lower_arm_right or 0)) / max(sum(1 for x in [lower_arm_left, lower_arm_right] if x), 1)
        if (lower_arm_left or lower_arm_right) else None,
    )
    m["upper_to_lower_leg_ratio"] = ratio(
        ((upper_leg_left or 0) + (upper_leg_right or 0)) / max(sum(1 for x in [upper_leg_left, upper_leg_right] if x), 1)
        if (upper_leg_left or upper_leg_right) else None,
        ((lower_leg_left or 0) + (lower_leg_right or 0)) / max(sum(1 for x in [lower_leg_left, lower_leg_right] if x), 1)
        if (lower_leg_left or lower_leg_right) else None,
    )

    # ------------------------------------------------------------------
    # 10. Posture indicators
    # ------------------------------------------------------------------
    # Shoulder level difference (positive = left shoulder higher in image = right shoulder anatomically)
    shoulder_level_diff = None
    if ls and rs:
        # In image coords y increases downward; higher y = lower in frame
        shoulder_level_diff = round(abs(ls["y"] - rs["y"]) * (cm_per_unit or 1), 4 if cm_per_unit is None else 1)
    m["shoulder_level_diff_cm"] = shoulder_level_diff if cm_per_unit else None
    m["shoulder_level_diff_norm"] = round(abs(ls["y"] - rs["y"]), 6) if ls and rs else None

    # Hip level difference
    hip_level_diff_norm = None
    if lh and rh:
        hip_level_diff_norm = round(abs(lh["y"] - rh["y"]), 6)
    m["hip_level_diff_norm"] = hip_level_diff_norm
    m["hip_level_diff_cm"]   = to_cm(hip_level_diff_norm)

    # Lateral trunk lean: angle of shoulder-midpoint→hip-midpoint line from vertical
    trunk_lean_deg = None
    if shoulder_mid and hip_mid:
        dx = hip_mid["x"] - shoulder_mid["x"]
        dy = hip_mid["y"] - shoulder_mid["y"]
        # Angle from vertical (pure vertical = 0°)
        trunk_lean_deg = round(math.degrees(math.atan2(dx, dy)), 2)
    m["trunk_lean_deg"] = trunk_lean_deg

    # Coverage summary
    coverage = body_pose.get("coverage", {})

    return {
        "analyzer_version": ANALYZER_VERSION,
        "analyzed_at":      datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "method":              "iod_population_mean" if cm_per_unit else "none",
            "iod_calibration_mm":  IOD_CALIBRATION_MM if cm_per_unit else None,
            "cm_per_norm_unit":    round(cm_per_unit, 4) if cm_per_unit else None,
        },
        "coverage":     coverage,
        "measurements": m,
    }


def apply_body_geometry(artifact_path: Path) -> dict:
    """
    Compute and write body_geometry block into an existing artifact JSON.
    Returns the updated artifact dict.
    """
    artifact = json.loads(artifact_path.read_text())
    artifact["body_geometry"] = analyze_body(artifact)
    artifact_path.write_text(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.body_analyzer <artifact.json> [artifact2.json ...]")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        p = Path(path_str)
        print(f"Analyzing {p.name}...", end=" ", flush=True)
        try:
            updated = apply_body_geometry(p)
            g = updated["body_geometry"]
            m = g["measurements"]
            h = m.get("height_cm", "?")
            sw = m.get("shoulder_width_cm", "?")
            hw = m.get("hip_width_cm", "?")
            print(f"OK  height={h}cm  shoulder={sw}cm  hip={hw}cm")
        except ValueError as e:
            print(f"SKIP  {e}")
