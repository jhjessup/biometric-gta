"""
Geometry Analyzer
Derives biometric facial measurements from stored MediaPipe landmark coordinates.
Operates entirely offline on existing artifact JSON — no image required.

All linear measurements are normalized by inter-ocular distance (IOD),
defined as the distance between left and right inner eye canthi (landmarks 133, 362).
This makes measurements pose- and scale-invariant.

MediaPipe FaceLandmarker 478-point canonical landmark indices used here
are validated against the canonical face model topology.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ANALYZER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Canonical MediaPipe FaceLandmarker landmark indices
# ---------------------------------------------------------------------------
LM = {
    # --- Eyes ---
    "left_eye_outer":       33,     # outer canthus (temporal), subject's right
    "left_eye_inner":       133,    # inner canthus (nasal), subject's right
    "left_eye_top":         159,    # upper lid apex
    "left_eye_bottom":      145,    # lower lid nadir
    "right_eye_outer":      263,    # outer canthus (temporal), subject's left
    "right_eye_inner":      362,    # inner canthus (nasal), subject's left
    "right_eye_top":        386,    # upper lid apex
    "right_eye_bottom":     374,    # lower lid nadir

    # --- Eyebrows ---
    "brow_left_inner":      107,    # medial end of left brow
    "brow_left_peak":        70,    # arch peak of left brow
    "brow_left_outer":       55,    # lateral end of left brow
    "brow_right_inner":     336,    # medial end of right brow
    "brow_right_peak":      300,    # arch peak of right brow
    "brow_right_outer":     285,    # lateral end of right brow

    # --- Nose ---
    "nasion":               168,    # root of nose / bridge between eyes
    "nose_tip":               4,    # pronasale (most anterior nose point)
    "nose_base":             94,    # base of columella / subnasale area
    "left_alar":            129,    # left alar base (nostril outer edge)
    "right_alar":           358,    # right alar base (nostril outer edge)

    # --- Mouth & Lips ---
    "mouth_left":            61,    # left oral commissure
    "mouth_right":          291,    # right oral commissure
    "upper_lip_top":          0,    # superior vermillion border, upper lip center
    "upper_lip_bottom":      13,    # inferior border of upper lip (inner)
    "lower_lip_top":         14,    # superior border of lower lip (inner)
    "lower_lip_bottom":      17,    # inferior vermillion border, lower lip center

    # --- Jaw & Chin ---
    "chin":                 152,    # menton (lowest chin point)
    "jaw_left":             172,    # left gonion (jaw angle)
    "jaw_right":            397,    # right gonion (jaw angle)

    # --- Face Width (cheekbones) ---
    "zygion_left":          234,    # left zygion (widest cheekbone point)
    "zygion_right":         454,    # right zygion (widest cheekbone point)

    # --- Forehead ---
    "forehead":              10,    # glabella / forehead center
}


def _dist2d(p1: dict, p2: dict) -> float:
    """Euclidean distance in the xy image plane."""
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def _midpoint(p1: dict, p2: dict) -> dict:
    return {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2, "z": 0}


def analyze_geometry(artifact: dict) -> dict:
    """
    Compute biometric facial geometry from a GTA artifact dict.

    Args:
        artifact: Parsed artifact JSON containing landmarks.face_mesh

    Returns:
        geometry dict suitable for artifact['geometry']
    """
    mesh = {lm["index"]: lm for lm in artifact["landmarks"]["face_mesh"]}

    def pt(name: str) -> dict:
        return mesh[LM[name]]

    # ------------------------------------------------------------------
    # Normalization baseline: inter-ocular distance (inner canthi)
    # ------------------------------------------------------------------
    iod_raw = _dist2d(pt("left_eye_inner"), pt("right_eye_inner"))

    def norm(d: float) -> float:
        """Normalize a distance by IOD, rounded to 4 dp."""
        return round(d / iod_raw, 4) if iod_raw > 0 else None

    def ratio(a: float, b: float) -> float:
        return round(a / b, 4) if b > 0 else None

    m = {}   # measurements dict

    # ------------------------------------------------------------------
    # 1. Eye measurements
    # ------------------------------------------------------------------
    l_eye_w  = _dist2d(pt("left_eye_outer"),  pt("left_eye_inner"))
    l_eye_h  = _dist2d(pt("left_eye_top"),    pt("left_eye_bottom"))
    r_eye_w  = _dist2d(pt("right_eye_outer"), pt("right_eye_inner"))
    r_eye_h  = _dist2d(pt("right_eye_top"),   pt("right_eye_bottom"))

    m["eye_left_width_norm"]       = norm(l_eye_w)
    m["eye_left_height_norm"]      = norm(l_eye_h)
    m["eye_left_aspect_ratio"]     = ratio(l_eye_h, l_eye_w)   # orbital index
    m["eye_right_width_norm"]      = norm(r_eye_w)
    m["eye_right_height_norm"]     = norm(r_eye_h)
    m["eye_right_aspect_ratio"]    = ratio(r_eye_h, r_eye_w)
    m["eye_width_symmetry"]        = ratio(l_eye_w, r_eye_w)   # 1.0 = perfect
    m["eye_height_symmetry"]       = ratio(l_eye_h, r_eye_h)

    # Inter-canthal distances
    outer_canthal = _dist2d(pt("left_eye_outer"), pt("right_eye_outer"))
    m["iod_inner_canthal_norm"]    = 1.0                        # by definition
    m["outer_canthal_dist_norm"]   = norm(outer_canthal)
    m["canthal_index"]             = ratio(iod_raw, outer_canthal)  # inner/outer

    # Eye level asymmetry (y-difference of upper lid apices)
    l_eye_center_y = (pt("left_eye_top")["y"]  + pt("left_eye_bottom")["y"])  / 2
    r_eye_center_y = (pt("right_eye_top")["y"] + pt("right_eye_bottom")["y"]) / 2
    m["eye_level_asymmetry_norm"]  = norm(abs(l_eye_center_y - r_eye_center_y))

    # Canthal tilt: angle of line connecting outer to inner canthus (each eye)
    def _angle_deg(p1, p2):
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        return round(math.degrees(math.atan2(dy, dx)), 2)

    m["canthal_tilt_left_deg"]     = _angle_deg(pt("left_eye_outer"),  pt("left_eye_inner"))
    m["canthal_tilt_right_deg"]    = _angle_deg(pt("right_eye_outer"), pt("right_eye_inner"))

    # ------------------------------------------------------------------
    # 2. Eyebrow measurements
    # ------------------------------------------------------------------
    l_brow_w = _dist2d(pt("brow_left_outer"),  pt("brow_left_inner"))
    r_brow_w = _dist2d(pt("brow_right_outer"), pt("brow_right_inner"))

    # Brow-to-eye vertical gap (positive = brow above eye top)
    l_brow_gap = abs(pt("brow_left_peak")["y"]  - pt("left_eye_top")["y"])
    r_brow_gap = abs(pt("brow_right_peak")["y"] - pt("right_eye_top")["y"])

    m["brow_left_width_norm"]      = norm(l_brow_w)
    m["brow_right_width_norm"]     = norm(r_brow_w)
    m["brow_width_symmetry"]       = ratio(l_brow_w, r_brow_w)
    m["brow_left_eye_gap_norm"]    = norm(l_brow_gap)
    m["brow_right_eye_gap_norm"]   = norm(r_brow_gap)
    m["brow_gap_symmetry"]         = ratio(l_brow_gap, r_brow_gap)

    # ------------------------------------------------------------------
    # 3. Nose measurements
    # ------------------------------------------------------------------
    nose_width  = _dist2d(pt("left_alar"),  pt("right_alar"))
    nose_length = _dist2d(pt("nasion"),     pt("nose_tip"))
    nose_bridge = _dist2d(pt("nasion"),     pt("nose_base"))

    # Nose tip deviation from midline
    face_mid_x  = (pt("zygion_left")["x"] + pt("zygion_right")["x"]) / 2
    nose_dev    = abs(pt("nose_tip")["x"] - face_mid_x)

    m["nose_width_norm"]           = norm(nose_width)
    m["nose_length_norm"]          = norm(nose_length)
    m["nose_bridge_length_norm"]   = norm(nose_bridge)
    m["nasal_index"]               = round(ratio(nose_width, nose_length) * 100, 2) if nose_length > 0 else None
    m["nose_tip_deviation_norm"]   = norm(nose_dev)
    m["alar_to_iod_ratio"]         = ratio(nose_width, iod_raw)

    # ------------------------------------------------------------------
    # 4. Mouth & lip measurements
    # ------------------------------------------------------------------
    mouth_w       = _dist2d(pt("mouth_left"),       pt("mouth_right"))
    upper_lip_h   = _dist2d(pt("upper_lip_top"),    pt("upper_lip_bottom"))
    lower_lip_h   = _dist2d(pt("lower_lip_top"),    pt("lower_lip_bottom"))
    philtrum      = _dist2d(pt("nose_base"),         pt("upper_lip_top"))

    mouth_center  = _midpoint(pt("mouth_left"), pt("mouth_right"))
    mouth_to_chin = abs(pt("chin")["y"] - mouth_center["y"])
    nose_to_mouth = abs(mouth_center["y"] - pt("nose_base")["y"])

    m["mouth_width_norm"]          = norm(mouth_w)
    m["mouth_to_iod_ratio"]        = ratio(mouth_w, iod_raw)
    m["upper_lip_height_norm"]     = norm(upper_lip_h)
    m["lower_lip_height_norm"]     = norm(lower_lip_h)
    m["lip_ratio"]                 = ratio(upper_lip_h, lower_lip_h)
    m["philtrum_length_norm"]      = norm(philtrum)
    m["mouth_to_chin_norm"]        = norm(mouth_to_chin)
    m["nose_to_mouth_norm"]        = norm(nose_to_mouth)

    # Mouth deviation from midline
    mouth_dev = abs(mouth_center["x"] - face_mid_x)
    m["mouth_deviation_norm"]      = norm(mouth_dev)

    # ------------------------------------------------------------------
    # 5. Face structure & proportions
    # ------------------------------------------------------------------
    face_w  = _dist2d(pt("zygion_left"), pt("zygion_right"))
    face_h  = _dist2d(pt("forehead"),    pt("chin"))
    jaw_w   = _dist2d(pt("jaw_left"),    pt("jaw_right"))

    m["face_width_norm"]           = norm(face_w)
    m["face_height_norm"]          = norm(face_h)
    m["jaw_width_norm"]            = norm(jaw_w)
    m["facial_index"]              = round(ratio(face_h, face_w) * 100, 2) if face_w > 0 else None
    m["jaw_to_cheek_ratio"]        = ratio(jaw_w, face_w)
    m["face_to_iod_ratio"]         = ratio(face_w, iod_raw)

    # ------------------------------------------------------------------
    # 6. Facial thirds (vertical proportions)
    # Upper:  forehead → nasion
    # Middle: nasion → nose_base
    # Lower:  nose_base → chin
    # ------------------------------------------------------------------
    upper_t = _dist2d(pt("forehead"),  pt("nasion"))
    mid_t   = _dist2d(pt("nasion"),    pt("nose_base"))
    lower_t = _dist2d(pt("nose_base"), pt("chin"))
    total_t = upper_t + mid_t + lower_t

    m["facial_third_upper"]        = round(upper_t / total_t, 4) if total_t > 0 else None
    m["facial_third_mid"]          = round(mid_t   / total_t, 4) if total_t > 0 else None
    m["facial_third_lower"]        = round(lower_t / total_t, 4) if total_t > 0 else None

    # ------------------------------------------------------------------
    # 7. Symmetry index (overall)
    # Average of key left-right asymmetry values; lower = more symmetric
    # ------------------------------------------------------------------
    asym_values = [v for v in [
        m.get("eye_level_asymmetry_norm"),
        m.get("nose_tip_deviation_norm"),
        m.get("mouth_deviation_norm"),
        abs(1.0 - m["eye_width_symmetry"])  if m.get("eye_width_symmetry")  else None,
        abs(1.0 - m["brow_width_symmetry"]) if m.get("brow_width_symmetry") else None,
    ] if v is not None]
    m["symmetry_index"] = round(sum(asym_values) / len(asym_values), 4) if asym_values else None

    return {
        "analyzer_version": ANALYZER_VERSION,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "iod_raw": round(iod_raw, 6),
        "measurements": m,
    }


def apply_geometry(artifact_path: Path) -> dict:
    """
    Compute and write geometry block into an existing artifact JSON.
    Returns the updated artifact dict.
    """
    artifact = json.loads(artifact_path.read_text())
    artifact["geometry"] = analyze_geometry(artifact)
    artifact_path.write_text(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.geometry_analyzer <artifact.json> [artifact2.json ...]")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        p = Path(path_str)
        print(f"Analyzing {p.name}...", end=" ", flush=True)
        updated = apply_geometry(p)
        g = updated["geometry"]
        m = g["measurements"]
        print(f"OK  facial_index={m['facial_index']}  nasal_index={m['nasal_index']}  symmetry={m['symmetry_index']}")
