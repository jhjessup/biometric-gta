"""
OpenRouter Image Generation Client
Drop-in generation backend for the biometric calibration pipeline.

Uses the OpenRouter API (OpenAI-compatible) to generate images via FLUX or other models.

Usage (library — drop-in for perchance_http_client.run_generation):
    from scripts.openrouter_client import run_generation
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")


def run_generation(
    prompt_data: dict,
    out_dir: Path,
    batch_size: int = 3,
    seed: int | None = None,
    generator_url: str | None = None,   # unused — kept for signature compat
    headless: bool = True,              # unused — kept for signature compat
    guidance_scale: float = 7.0,        # stored in metadata; FLUX doesn't use CFG same way
    model: str = "qwen/qwen3.6-plus-preview",
    resolution: str = "1024x1024",
    **kwargs,                           # absorbs reference_image_path, ip_adapter_scale, etc.
) -> list[Path]:
    """
    Generate images via OpenRouter and save them to out_dir.

    Args:
        prompt_data: dict with keys:
            - positive_prompt (str, required)
            - negative_prompt (str, optional)
            - style_selector (str, optional, unused by this backend)
        out_dir: directory where images and metadata.json are written
        batch_size: number of images to generate
        seed: optional RNG seed (passed to OpenRouter if provided)
        generator_url: unused, kept for signature compatibility
        headless: unused, kept for signature compatibility
        guidance_scale: stored in metadata only; FLUX doesn't expose CFG the same way
        model: OpenRouter model identifier
        resolution: image size string, e.g. "1024x1024"
        **kwargs: absorbs extra args (reference_image_path, ip_adapter_scale, etc.)

    Returns:
        List of Path objects for the saved PNG images.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_prompt: str = prompt_data.get("positive_prompt", "")
    negative_prompt: str | None = prompt_data.get("negative_prompt")

    # OpenRouter image models don't have a separate negative_prompt field in the
    # standard API — append it to the positive prompt instead.
    full_prompt = positive_prompt
    if negative_prompt:
        full_prompt = f"{positive_prompt} , avoid: {negative_prompt}"

    saved_paths: list[Path] = []

    for i in range(batch_size):
        filename = out_dir / f"openrouter_{i:03d}.png"
        try:
            extra_body = {"seed": seed} if seed is not None else {}
            response = client.images.generate(
                model=model,
                prompt=full_prompt,
                n=1,
                size=resolution,
                response_format="b64_json",
                extra_body=extra_body,
            )
            image_b64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_b64)
            filename.write_bytes(image_bytes)
            saved_paths.append(filename)
        except Exception as exc:
            print(
                f"[openrouter_client] Error generating image {i} ({filename.name}): {exc}",
                file=sys.stderr,
            )

    metadata = {
        "prompt": full_prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "resolution": resolution,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_generated": len(saved_paths),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return saved_paths
