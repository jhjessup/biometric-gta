"""
Hair Color Analyzer
Computes hair color from a stripped image using face landmark geometry.

Samples the region between the bounding box top and the forehead landmark
(index 10), converts to HSV, takes the median across sampled pixels, and
classifies into a named color + shade descriptor suitable for prompt building.

No external API calls — fully offline.

Usage:
    from agents.hair_analyzer import analyze_hair_color
    result = analyze_hair_color(image_path, artifact)
    # result: {"color_name": "medium_brown", "shade_descriptor": "warm medium brown",
    #          "rgb_median": [r, g, b], "hsv_median": [h, s, v], "confidence": 0.84}
"""

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

# Landmark indices
_FOREHEAD_IDX   = 10    # glabella / central forehead — best proxy for hairline
_BROW_LEFT_IDX  = 107   # inner left brow (defines lateral boundary)
_BROW_RIGHT_IDX = 336   # inner right brow


# ---------------------------------------------------------------------------
# Color classification
# ---------------------------------------------------------------------------

# Each rule: (h_lo, h_hi, s_lo, v_lo, v_hi, label, shade_descriptor)
# h/s/v all in [0, 1]  (colorsys output)
_COLOR_RULES = [
    # Achromatic — checked by s threshold first
    (0.00, 1.00, 0.00, 0.80, 1.00, "white",       "white"),
    (0.00, 1.00, 0.00, 0.55, 0.80, "gray_silver",  "silver gray"),
    (0.00, 1.00, 0.00, 0.35, 0.55, "dark_gray",    "dark gray"),
    # Chromatic — ordered light → dark within hue bands
    # Red / auburn  (hue wraps: 0-0.05 and 0.95-1.0)
    (0.95, 1.00, 0.25, 0.25, 1.00, "red_auburn",   "deep auburn"),
    (0.00, 0.06, 0.25, 0.25, 1.00, "red_auburn",   "deep auburn"),
    # Blonde / golden (warm yellowish hue, moderate-high value)
    (0.06, 0.17, 0.08, 0.72, 1.00, "light_blonde", "light golden blonde"),
    (0.06, 0.17, 0.15, 0.55, 0.72, "dark_blonde",  "dark blonde"),
    # Brown band (most common — sub-classify by value)
    (0.04, 0.14, 0.20, 0.50, 1.00, "light_brown",  "warm light brown"),
    (0.04, 0.14, 0.25, 0.30, 0.50, "medium_brown", "warm medium brown"),
    (0.04, 0.14, 0.20, 0.15, 0.30, "dark_brown",   "dark brown"),
    # Black
    (0.00, 1.00, 0.00, 0.00, 0.15, "black",        "black"),
]

_SATURATION_ACHROMATIC_THRESHOLD = 0.12  # below this → achromatic rules apply


def _classify(h: float, s: float, v: float) -> tuple[str, str]:
    """Map (h, s, v) ∈ [0,1]³ to (color_name, shade_descriptor)."""
    if s < _SATURATION_ACHROMATIC_THRESHOLD:
        # Use achromatic rules (first three in _COLOR_RULES)
        for h_lo, h_hi, s_lo, v_lo, v_hi, label, shade in _COLOR_RULES[:3]:
            if v_lo <= v <= v_hi:
                return label, shade
        return "dark_gray", "dark gray"

    for h_lo, h_hi, s_lo, v_lo, v_hi, label, shade in _COLOR_RULES[3:]:
        if h_lo <= h <= h_hi and s >= s_lo and v_lo <= v <= v_hi:
            return label, shade

    # Fallback by value
    if v < 0.20:
        return "black", "black"
    if v < 0.40:
        return "dark_brown", "dark brown"
    if v < 0.60:
        return "medium_brown", "medium brown"
    return "light_brown", "light brown"


# ---------------------------------------------------------------------------
# Region sampling
# ---------------------------------------------------------------------------

def _hair_region_pixels(image: Image.Image, artifact: dict) -> np.ndarray | None:
    """
    Extract pixel array (N×3 RGB uint8) from the hair region above the forehead.

    Returns None if the region is too small to be reliable.
    """
    w, h = image.size

    mesh = {lm["index"]: lm for lm in artifact["landmarks"]["face_mesh"]}
    bb   = artifact["landmarks"]["bounding_box"]

    forehead = mesh.get(_FOREHEAD_IDX)
    if forehead is None:
        return None

    # The bounding box top sits flush with the forehead landmark —
    # hair is above the detected face region. Sample a band from
    # 20% of image height above the forehead down to 3% above it.
    y_bottom = int(max(0, forehead["y"] - 0.03) * h)
    y_top    = int(max(0, forehead["y"] - 0.20) * h)

    # Lateral bounds: full bounding box width
    x_left  = int(bb["x_min"] * w)
    x_right = int(bb["x_max"] * w)

    # Clamp to image bounds
    y_top    = max(0, y_top)
    y_bottom = min(h - 1, y_bottom)
    x_left   = max(0, x_left)
    x_right  = min(w - 1, x_right)

    region_h = y_bottom - y_top
    region_w = x_right  - x_left

    if region_h < 5 or region_w < 5:
        return None

    crop = image.crop((x_left, y_top, x_right, y_bottom)).convert("RGB")
    arr  = np.array(crop).reshape(-1, 3)   # (N, 3) uint8

    # Filter near-white pixels (blown-out highlights / background bleed)
    brightness = arr.mean(axis=1)
    mask = brightness < 230
    arr  = arr[mask]

    # Filter near-black pixels (deep shadows)
    brightness = brightness[mask]
    arr = arr[brightness > 15]

    if len(arr) < 20:
        return None

    # Filter low-saturation (achromatic) pixels — removes white/gray backgrounds
    # and leaves only chromatic hair pixels.
    hsv_s = np.array([
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[1]
        for r, g, b in arr
    ])
    chromatic = arr[hsv_s > 0.08]

    # If we have enough chromatic pixels, use them; otherwise fall back to all pixels
    # (handles naturally gray/white hair which has low saturation by definition)
    if len(chromatic) >= 30:
        return chromatic

    return arr


def _confidence(pixels: np.ndarray) -> float:
    """
    Rough confidence score based on sample size and color consistency.
    More pixels + lower std-dev in HSV value channel → higher confidence.
    """
    n = len(pixels)
    size_score = min(1.0, n / 500)

    hsv_vals = np.array([
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[2]
        for r, g, b in pixels[:200]   # sample up to 200 for speed
    ])
    consistency_score = max(0.0, 1.0 - (hsv_vals.std() / 0.35))

    return round(float(size_score * 0.4 + consistency_score * 0.6), 3)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyze_hair_color(image_path: Path, artifact: dict) -> dict:
    """
    Analyze hair color using landmark-guided region sampling.

    Args:
        image_path: Path to the stripped (EXIF-free) source image.
        artifact:   GTA artifact dict (must contain landmarks.face_mesh and
                    landmarks.bounding_box).

    Returns:
        Dict with keys:
            color_name        — canonical label (e.g. "medium_brown")
            shade_descriptor  — prompt-ready phrase (e.g. "warm medium brown")
            rgb_median        — [R, G, B] uint8 median of sampled pixels
            hsv_median        — [H, S, V] float ∈ [0,1]
            sample_size       — number of pixels sampled
            confidence        — float 0–1 (low = small/inconsistent region)
    """
    image  = Image.open(image_path).convert("RGB")
    pixels = _hair_region_pixels(image, artifact)

    if pixels is None or len(pixels) == 0:
        return {
            "color_name":       "unknown",
            "shade_descriptor": "unknown",
            "rgb_median":       None,
            "hsv_median":       None,
            "sample_size":      0,
            "confidence":       0.0,
        }

    rgb_med = np.median(pixels, axis=0).astype(int).tolist()
    h, s, v = colorsys.rgb_to_hsv(rgb_med[0] / 255, rgb_med[1] / 255, rgb_med[2] / 255)

    color_name, shade_descriptor = _classify(h, s, v)
    conf = _confidence(pixels)

    return {
        "color_name":       color_name,
        "shade_descriptor": shade_descriptor,
        "rgb_median":       rgb_med,
        "hsv_median":       [round(h, 4), round(s, 4), round(v, 4)],
        "sample_size":      len(pixels),
        "confidence":       conf,
    }
