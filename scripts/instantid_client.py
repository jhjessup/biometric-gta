"""
InstantID Client
Identity-conditioned image generation via the Kaggle InstantID server.

Drop-in replacement for perchance_http_client.py — same run_generation()
signature with one additional parameter: reference_image_path.

The server (notebooks/kaggle_instantid_server.ipynb) must be running and
INSTANTID_SERVER_URL must be set in .env.
"""

import base64
import io
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

DEFAULT_SERVER_URL = os.getenv("INSTANTID_SERVER_URL", "").rstrip("/")


def run_generation(
    prompt_data: dict,
    out_dir: Path,
    batch_size: int = 3,
    seed: int | None = None,
    generator_url: str | None = None,
    headless: bool = True,          # kept for signature compatibility
    guidance_scale: float = 7.0,
    reference_image_path: Path | None = None,
    ip_adapter_scale: float = 0.8,
) -> list[Path]:
    """
    Generate identity-conditioned images via the InstantID server.

    Args:
        prompt_data:          Output from agents.prompt_builder.build_prompt()
        out_dir:              Directory to save generated images
        batch_size:           Number of images to generate per call
        seed:                 RNG seed for reproducibility (None = random each call)
        generator_url:        Server URL; overrides INSTANTID_SERVER_URL env var
        headless:             Ignored (kept for drop-in compatibility)
        guidance_scale:       CFG scale (7.0 is the validated optimum for this subject)
        reference_image_path: EXIF-stripped source image for face identity conditioning
        ip_adapter_scale:     InstantID identity strength, 0.0–1.0.
                              0.8 = strong identity, some generative flexibility.
                              1.0 = maximum identity lock, less variation.

    Returns:
        List of Paths to saved PNG images.

    Raises:
        ValueError:   reference_image_path not provided or server URL not configured
        RuntimeError: server unreachable or returned an error
    """
    server_url = (generator_url or DEFAULT_SERVER_URL)
    if not server_url:
        raise ValueError(
            "INSTANTID_SERVER_URL is not set. "
            "Start the Kaggle notebook and add the printed URL to .env."
        )

    if reference_image_path is None:
        raise ValueError(
            "reference_image_path is required. "
            "Pass the EXIF-stripped source image for the target subject, or use "
            "agents.reference_selector.select_reference() to pick the best one."
        )
    reference_image_path = Path(reference_image_path)
    if not reference_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

    # Health check before sending large payload
    try:
        resp = requests.get(f"{server_url}/health", timeout=15)
        resp.raise_for_status()
        health = resp.json()
        if not health.get("gpu"):
            print("WARNING: server reports no GPU — generation will be slow")
    except requests.RequestException as exc:
        raise RuntimeError(
            f"InstantID server unreachable at {server_url}. "
            f"Verify the Kaggle notebook is still running. Error: {exc}"
        ) from exc

    # Encode reference face image
    face_img = Image.open(reference_image_path).convert("RGB")
    buf = io.BytesIO()
    face_img.save(buf, format="PNG")
    face_b64 = base64.b64encode(buf.getvalue()).decode()

    payload: dict = {
        "positive_prompt":  prompt_data["positive_prompt"],
        "negative_prompt":  prompt_data.get("negative_prompt", ""),
        "face_image_b64":   face_b64,
        "num_images":       batch_size,
        "guidance_scale":   guidance_scale,
        "ip_adapter_scale": ip_adapter_scale,
    }
    if seed is not None:
        payload["seed"] = seed

    print(f"Generating {batch_size} image(s) via InstantID ({server_url})...")
    try:
        resp = requests.post(
            f"{server_url}/generate",
            json=payload,
            timeout=600,   # T4 generates ~30s/image; allow headroom for batches
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Generation request failed: {exc}") from exc

    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"Server error: {result['error']}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for i, img_b64 in enumerate(result["images"]):
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        dest = out_dir / f"instantid_{i:03d}.png"
        img.save(dest)
        saved.append(dest)
        print(f"  Saved: {dest.name}")

    return saved
