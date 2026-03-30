"""
Perchance Prompt Builder
Translates a GTA artifact into a weighted Stable Diffusion prompt suitable
for the perchance.org AI image generator.

Produces two strings:
  - positive_prompt: full subject description with biometric weighting
  - negative_prompt: exclusion terms to suppress artifacts and drift

Design principles:
  - Style selector is treated as a constant (set externally to "Professional Photo")
  - All biometric tuning uses (keyword:weight) notation; weight range 0.9–1.3
  - Subject core drawn from enrichment.forensic (Gemini-derived)
  - Biometric tuning drawn from geometry.measurements (MediaPipe-derived)
  - Body drawn from body_geometry.measurements where available
  - Sartorial drawn from enrichment.sartorial for clothing reproduction
  - Missing data degrades gracefully — slots are omitted rather than hallucinated
"""

import json
from pathlib import Path

BUILDER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Biometric descriptor lookup tables
# Maps measurement ranges → prompt tokens + weights
# ---------------------------------------------------------------------------

# facial_index = face_height / face_width * 100
# Higher = longer/narrower face; lower = wider/rounder face
FACIAL_INDEX_DESCRIPTORS = [
    (0,   82,  "very wide round face",       1.2),
    (82,  90,  "wide face",                  1.1),
    (90,  100, "balanced facial proportions", 1.0),
    (100, 110, "oval face shape",             1.0),
    (110, 120, "long oval face",              1.1),
    (120, 999, "long narrow face",            1.2),
]

# nasal_index = nose_width / nose_length * 100
NASAL_INDEX_DESCRIPTORS = [
    (0,   70,  "narrow refined nose",   1.1),
    (70,  90,  "straight nose",         1.0),
    (90,  110, "broad nose",            1.1),
    (110, 999, "wide flat nose",        1.2),
]

# jaw_to_cheek_ratio
JAW_DESCRIPTORS = [
    (0,    0.68, "defined cheekbones, tapered jawline", 1.1),
    (0.68, 0.80, "balanced jawline",                    1.0),
    (0.80, 0.90, "strong square jawline",               1.1),
    (0.90, 999,  "wide angular jaw",                    1.2),
]

# canthal_tilt (mean of left and right, degrees)
# Positive = upturned (outer corner higher than inner)
# In image coords (y increases downward): upturned = outer_y > inner_y → positive atan2
CANTHAL_DESCRIPTORS = [
    (-999, -8,  "downturned eyes",  1.2),
    (-8,   -3,  "slightly downturned eyes", 1.1),
    (-3,    3,  "neutral eye angle", 1.0),
    (3,     8,  "slightly upturned eyes", 1.1),
    (8,   999,  "upturned almond eyes", 1.2),
]

# eye aspect ratio (height/width) — average of both eyes
EYE_ASPECT_DESCRIPTORS = [
    (0,     0.22, "narrow hooded eyes",  1.2),
    (0.22,  0.30, "almond-shaped eyes",  1.0),
    (0.30,  0.38, "open expressive eyes", 1.1),
    (0.38,  999,  "large wide eyes",     1.2),
]

# symmetry_index — lower is more symmetric
SYMMETRY_DESCRIPTORS = [
    (0,    0.10, None),                         # highly symmetric — don't add token
    (0.10, 0.25, None),                         # symmetric — don't add token
    (0.25, 0.45, "subtle facial asymmetry", 0.9),
    (0.45, 999,  "natural facial asymmetry", 1.0),
]

# facial_third_lower proportion (nose-base to chin / total face height)
# Higher lower-third = longer chin/lower face
LOWER_THIRD_DESCRIPTORS = [
    (0,     0.32, "short chin",        1.1),
    (0.32,  0.42, "balanced chin",     1.0),
    (0.42,  999,  "prominent long chin", 1.1),
]

# shoulder_to_hip_ratio — body shape proxy
BODY_SHAPE_DESCRIPTORS = [
    (0,    0.88, "pear-shaped figure, narrow shoulders, wider hips", 1.1),
    (0.88, 1.05, "balanced hourglass figure",                        1.0),
    (1.05, 1.20, "athletic figure, broad shoulders",                 1.1),
    (1.20, 999,  "inverted triangle figure, wide shoulders",         1.2),
]


def _lookup(value: float | None, table: list) -> tuple[str | None, float]:
    """Return (descriptor, weight) for a value from a lookup table."""
    if value is None:
        return None, 1.0
    for *bounds, descriptor, weight in table:
        lo, hi = bounds[0], bounds[1]
        if lo <= value < hi:
            return descriptor, weight
    return None, 1.0


def _lookup_no_weight(value: float | None, table: list) -> str | None:
    """For tables where some entries have no descriptor (None)."""
    if value is None:
        return None
    for entry in table:
        lo, hi = entry[0], entry[1]
        descriptor = entry[2] if len(entry) > 2 else None
        weight = entry[3] if len(entry) > 3 else 1.0
        if lo <= value < hi:
            return (descriptor, weight) if descriptor else None
    return None


def _weighted_token(descriptor: str | None, weight: float) -> str | None:
    if descriptor is None:
        return None
    if abs(weight - 1.0) < 0.05:
        return descriptor
    return f"({descriptor}:{weight})"


def _hair_length_style_prefix(forensic: dict) -> str:
    """Return 'long loose ' style prefix for use with Python-computed color."""
    length = forensic.get("hair_length", "")
    style  = forensic.get("hair_style", "")
    parts  = []
    if length and length not in ("unknown", "bald"):
        parts.append(length)
    if style and style not in ("unknown", "loose"):
        parts.append(style)
    return (" ".join(parts) + " ") if parts else ""


def _hair_description(forensic: dict) -> str | None:
    color  = forensic.get("hair_color")
    length = forensic.get("hair_length")
    style  = forensic.get("hair_style")
    if not color or color == "unknown":
        return None
    parts = []
    if length and length not in ("unknown", "bald"):
        parts.append(length)
    if style and style not in ("unknown", "loose"):
        parts.append(style)
    if color != "unknown":
        parts.append(color)
    parts.append("hair")
    return " ".join(parts)


def _age_gender_phrase(forensic: dict) -> str:
    age = forensic.get("estimated_age_range", "")
    if not age or age == "unknown":
        return "woman"
    # Parse range like "20-30" → midpoint label
    try:
        lo, hi = (int(x) for x in age.split("-"))
        mid = (lo + hi) // 2
        if mid < 25:
            return "young woman in her early twenties"
        elif mid < 35:
            return "woman in her late twenties to early thirties"
        elif mid < 45:
            return "woman in her late thirties to early forties"
        else:
            return "woman in her mid forties"
    except Exception:
        return f"woman aged {age}"


def _sartorial_description(sartorial: dict) -> list[str]:
    tokens = []
    top = sartorial.get("top_garment", "unknown")
    top_color = sartorial.get("top_color", "")
    top_pattern = sartorial.get("top_pattern", "solid")
    if top and top not in ("unknown", "none_visible"):
        desc = f"{top_color} {top}" if top_color and top_color != "unknown" else top
        if top_pattern and top_pattern not in ("unknown", "none_visible", "solid"):
            desc = f"{top_pattern} {desc}"
        details = sartorial.get("notable_details", [])
        if details:
            desc += f" with {', '.join(details)}"
        tokens.append(desc)
    bottom = sartorial.get("bottom_garment", "unknown")
    bottom_color = sartorial.get("bottom_color", "")
    if bottom and bottom not in ("unknown", "none_visible"):
        tokens.append(f"{bottom_color} {bottom}".strip() if bottom_color and bottom_color != "unknown" else bottom)
    accessories = sartorial.get("accessories", [])
    if accessories:
        tokens.append(", ".join(accessories))
    return tokens


def build_prompt(artifact: dict) -> dict:
    """
    Build a perchance-ready prompt pair from a GTA artifact.

    Returns:
        {
            "positive_prompt": str,
            "negative_prompt": str,
            "style_selector":  str,   # set this in the perchance Style dropdown
            "seed":            None,  # populated after first generation
            "builder_version": str,
            "source_artifact": str,
            "slot_diagnostics": dict  # raw values used for each slot
        }
    """
    enrichment = artifact.get("enrichment", {})
    forensic   = enrichment.get("forensic", {})
    sartorial  = enrichment.get("sartorial", {})
    geo       = artifact.get("geometry", {}).get("measurements", {})
    body      = artifact.get("body_geometry", {}).get("measurements", {})

    diag = {}  # slot diagnostics for calibration feedback

    # ------------------------------------------------------------------
    # Block 1: Subject core (from enrichment)
    # ------------------------------------------------------------------
    age_phrase  = _age_gender_phrase(forensic)
    skin        = forensic.get("skin_tone", "")
    eye_color   = forensic.get("eye_color", "")
    # Prefer Python-computed shade_descriptor (more precise than Gemini's coarse label)
    hair_analysis = enrichment.get("hair_analysis", {})
    if hair_analysis.get("shade_descriptor") and hair_analysis.get("confidence", 0) >= 0.60 and hair_analysis["shade_descriptor"] != "unknown":
        hair_desc = _hair_length_style_prefix(forensic) + hair_analysis["shade_descriptor"] + " hair"
    else:
        hair_desc = _hair_description(forensic)
    eyewear     = forensic.get("eyewear", "none")
    expression  = forensic.get("expression", "neutral")
    features    = forensic.get("distinctive_features", [])

    subject_tokens = [age_phrase]
    if skin and skin not in ("unknown",):
        subject_tokens.append(f"{skin} skin")
    if eye_color and eye_color not in ("unknown",):
        subject_tokens.append(f"{eye_color} eyes")
    if hair_desc:
        subject_tokens.append(hair_desc)
    if eyewear and eyewear not in ("none", "unknown"):
        subject_tokens.append(eyewear)
    if expression and expression not in ("neutral", "unknown"):
        subject_tokens.append(f"{expression} expression")
    if features:
        subject_tokens.extend(features)

    diag["subject_core"] = subject_tokens[:]

    # ------------------------------------------------------------------
    # Block 2: Biometric tuning (from geometry)
    # ------------------------------------------------------------------
    fi   = geo.get("facial_index")
    ni   = geo.get("nasal_index")
    jcr  = geo.get("jaw_to_cheek_ratio")
    tilt_l = geo.get("canthal_tilt_left_deg")
    tilt_r = geo.get("canthal_tilt_right_deg")
    tilt_mean = None
    if tilt_l is not None and tilt_r is not None:
        tilt_mean = (tilt_l + tilt_r) / 2
    ear_l = geo.get("eye_left_aspect_ratio")
    ear_r = geo.get("eye_right_aspect_ratio")
    ear_mean = None
    if ear_l is not None and ear_r is not None:
        ear_mean = (ear_l + ear_r) / 2
    sym  = geo.get("symmetry_index")
    lt   = geo.get("facial_third_lower")

    biometric_tokens = []

    for value, table, label in [
        (fi,       FACIAL_INDEX_DESCRIPTORS,  "facial_index"),
        (ni,       NASAL_INDEX_DESCRIPTORS,   "nasal_index"),
        (jcr,      JAW_DESCRIPTORS,           "jaw"),
        (tilt_mean,CANTHAL_DESCRIPTORS,       "canthal_tilt"),
        (ear_mean, EYE_ASPECT_DESCRIPTORS,    "eye_aspect"),
        (lt,       LOWER_THIRD_DESCRIPTORS,   "lower_third"),
    ]:
        if value is None:
            continue
        desc, weight = _lookup(value, table)
        token = _weighted_token(desc, weight)
        if token:
            biometric_tokens.append(token)
        diag[label] = {"value": value, "descriptor": desc, "weight": weight}

    # Symmetry (special — may produce no token)
    if sym is not None:
        result = _lookup_no_weight(sym, SYMMETRY_DESCRIPTORS)
        if result:
            desc, weight = result
            token = _weighted_token(desc, weight)
            if token:
                biometric_tokens.append(token)
        diag["symmetry"] = {"value": sym}

    # ------------------------------------------------------------------
    # Block 3: Body proportions (from body_geometry, if available)
    # ------------------------------------------------------------------
    body_tokens = []
    shr = body.get("shoulder_to_hip_ratio") if body else None
    if shr:
        desc, weight = _lookup(shr, BODY_SHAPE_DESCRIPTORS)
        token = _weighted_token(desc, weight)
        if token:
            body_tokens.append(token)
        diag["body_shape"] = {"value": shr, "descriptor": desc, "weight": weight}

    # ------------------------------------------------------------------
    # Block 4: Sartorial (from enrichment, optional)
    # ------------------------------------------------------------------
    sartorial_tokens = _sartorial_description(sartorial) if sartorial else []
    diag["sartorial"] = sartorial_tokens[:]

    # ------------------------------------------------------------------
    # Block 5: Optics / style constants
    # ------------------------------------------------------------------
    optics_tokens = [
        "portrait photography",
        "facing camera directly",
        "neutral background",
        "sharp focus",
        "high detail",
        "natural lighting",
        "photorealistic",
    ]

    # ------------------------------------------------------------------
    # Assemble positive prompt
    # ------------------------------------------------------------------
    sections = []
    if subject_tokens:
        sections.append(", ".join(subject_tokens))
    if biometric_tokens:
        sections.append(", ".join(biometric_tokens))
    if body_tokens:
            sections.append(", ".join(body_tokens))
    if sartorial_tokens:
        sections.append(", ".join(sartorial_tokens))
    sections.append(", ".join(optics_tokens))

    positive_prompt = ", ".join(sections)

    # ------------------------------------------------------------------
    # Negative prompt
    # ------------------------------------------------------------------
    negative_prompt = (
        "deformed, mutation, extra limbs, extra fingers, missing limbs, "
        "blurry, low quality, low resolution, jpeg artifacts, "
        "cartoon, anime, illustration, painting, drawing, sketch, "
        "text, watermark, signature, logo, "
        "disfigured, bad anatomy, bad proportions, "
        "unnatural skin, plastic skin, wax figure, "
        "multiple people, crowd, "
        "sunglasses" if eyewear in ("none", "unknown") else ""
    )

    return {
        "positive_prompt":  positive_prompt,
        "negative_prompt":  negative_prompt.strip(", "),
        "style_selector":   "Professional Photo",
        "aspect_ratio":     "Portrait",
        "seed":             None,
        "builder_version":  BUILDER_VERSION,
        "source_artifact":  artifact.get("artifact_id", "unknown"),
        "slot_diagnostics": diag,
    }


def build_from_file(artifact_path: Path) -> dict:
    artifact = json.loads(artifact_path.read_text())
    return build_prompt(artifact)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.prompt_builder <artifact.json>")
        sys.exit(1)

    result = build_from_file(Path(sys.argv[1]))
    print("=== POSITIVE PROMPT ===")
    print(result["positive_prompt"])
    print("\n=== NEGATIVE PROMPT ===")
    print(result["negative_prompt"])
    print("\n=== STYLE SELECTOR ===")
    print(result["style_selector"])
    print("\n=== SLOT DIAGNOSTICS ===")
    print(json.dumps(result["slot_diagnostics"], indent=2))
