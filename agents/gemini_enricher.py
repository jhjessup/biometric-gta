"""
Gemini Enrichment Agent
Sends EXIF-stripped images to Gemini for forensic biometric and sartorial analysis.
Appends structured enrichment data to existing GTA artifact JSON files.

Requires GEMINI_API_KEY in .env for standalone pipeline use.
Can also be driven externally (e.g., via MCP tool) by passing pre-fetched JSON.
"""

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

ENRICHMENT_MODEL = "gemini-2.5-flash"
ENRICH_MAX_SIDE = 320   # px — sufficient for attribute extraction, keeps tokens low

ENRICHMENT_PROMPT = """You are a forensic biometric analyst and fashion expert.
Analyze this image and return a single JSON object with exactly two top-level keys: "forensic" and "sartorial".

"forensic" must contain:
- "face_shape": string — one of: oval, round, square, heart, oblong, diamond, unknown
- "skin_tone": string — e.g. fair, light, medium, olive, tan, dark, unknown
- "eye_color": string — e.g. brown, blue, green, hazel, gray, unknown
- "hair_color": string — e.g. black, dark_brown, brown, auburn, blonde, red, gray, white, unknown
- "hair_length": string — one of: bald, very_short, short, medium, long, very_long, unknown
- "hair_style": string — e.g. straight, wavy, curly, braided, bun, ponytail, updo, loose, unknown
- "facial_hair": string — one of: none, stubble, mustache, goatee, beard, full_beard, unknown
- "estimated_age_range": string — e.g. "20-30", "35-45", unknown
- "distinctive_features": array of strings — scars, moles, freckles, birthmarks, piercings, tattoos visible. Empty array if none.
- "eyewear": string — one of: none, glasses, sunglasses, unknown
- "expression": string — one of: neutral, smiling, laughing, serious, other, unknown

"sartorial" must contain:
- "top_garment": string — e.g. t-shirt, blouse, shirt, jacket, hoodie, sweater, dress, top, none_visible, unknown
- "top_color": string — primary color, e.g. white, black, navy, red, unknown
- "top_pattern": string — one of: solid, striped, plaid, floral, graphic, logo, other, none_visible, unknown
- "bottom_garment": string — e.g. jeans, trousers, skirt, shorts, leggings, dress, none_visible, unknown
- "bottom_color": string — primary color or unknown
- "outerwear": string — one of: none, jacket, coat, blazer, cardigan, vest, other, unknown
- "accessories": array of strings — e.g. necklace, earrings, watch, bracelet, hat, bag, scarf. Empty if none visible.
- "footwear": string — one of: none_visible, sneakers, boots, heels, flats, sandals, loafers, other, unknown
- "style_category": string — one of: casual, smart_casual, formal, athletic, streetwear, bohemian, other, unknown
- "notable_details": array of strings — brand logos, distinctive patterns, unusual items. Empty if none.

Return ONLY the raw JSON object. No markdown, no explanation, no code fences."""


def _resize_for_gemini(image_path: Path) -> str:
    """Load, resize, and base64-encode an image for Gemini."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        scale = ENRICH_MAX_SIDE / max(w, h)
        if scale < 1.0:
            rgb = rgb.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()


def enrich_with_sdk(image_path: Path) -> dict:
    """
    Call Gemini via google-generativeai SDK. Requires GEMINI_API_KEY in .env.
    """
    import os
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(ENRICHMENT_MODEL)

    b64 = _resize_for_gemini(image_path)
    image_part = {"mime_type": "image/jpeg", "data": b64}

    response = model.generate_content(
        [ENRICHMENT_PROMPT, image_part],
        generation_config={"response_mime_type": "application/json"},
    )
    return json.loads(response.text)


def apply_enrichment(artifact_path: Path, enrichment: dict, model: str = ENRICHMENT_MODEL) -> dict:
    """
    Write enrichment data into an existing artifact JSON.
    Returns the updated artifact dict.
    """
    artifact = json.loads(artifact_path.read_text())
    artifact["enrichment"] = {
        "model": model,
        "enriched_at": datetime.now(timezone.utc).isoformat(),
        "forensic": enrichment.get("forensic", {}),
        "sartorial": enrichment.get("sartorial", {}),
    }
    artifact_path.write_text(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m agents.gemini_enricher <stripped_image> <artifact.json>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    artifact_path = Path(sys.argv[2])

    print(f"Enriching {artifact_path.name} via Gemini SDK...")
    enrichment = enrich_with_sdk(img_path)
    apply_enrichment(artifact_path, enrichment)
    print("Done.")
    print(json.dumps(enrichment, indent=2))
