"""
Perchance.org Browser Automation Driver
Uses Playwright to submit prompts to the perchance AI image generator,
wait for generation, and download outputs to a local directory.

Targets: https://perchance.org/ai-text-to-image-generator
         (or any perchance generator URL passed via --url)

Usage:
    python -m scripts.perchance_driver <prompt_json> [--out DIR] [--batch N] [--seed SEED] [--url URL]

    prompt_json: path to a JSON file produced by agents/prompt_builder.py
                 (must contain positive_prompt, negative_prompt, style_selector)

Requirements:
    pip install playwright
    playwright install chromium

Selector notes:
    Perchance generators are single-page apps. If selectors break after a
    site update, run:  python -m scripts.perchance_driver --dump-selectors --url <URL>
    to print all visible textarea/select/button elements for re-mapping.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Perchance UI selector map — adjust if the DOM changes
# ---------------------------------------------------------------------------
SELECTORS = {
    # Main prompt textarea (the large input box)
    "prompt_input":     'textarea[placeholder*="describe"], textarea.prompt, #prompt, textarea:first-of-type',
    # Negative prompt / anti-description textarea
    "negative_input":   'textarea[placeholder*="avoid"], textarea[placeholder*="negative"], textarea.negative-prompt, textarea:nth-of-type(2)',
    # Style dropdown (the one with 60+ presets)
    "style_select":     'select[id*="style"], select.style-select, select:first-of-type',
    # Generate / create button
    "generate_button":  'button[id*="generate"], button.generate, button:has-text("Generate"), button:has-text("Create")',
    # Seed input (may not exist in all generators)
    "seed_input":       'input[placeholder*="seed"], input[id*="seed"], input.seed',
    # Batch size control (may be a number input or stepper)
    "batch_input":      'input[id*="batch"], input[placeholder*="batch"]',
    # Generated image elements (images appear in a results grid)
    "result_images":    'img.generated-image, .image-grid img, .results img, img[src*="blob:"], img[src*="data:image"]',
}

# How long to wait for images to appear after clicking Generate (seconds)
GENERATION_TIMEOUT_S = 120


def _slug(text: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:max_len].strip("_")


def run_generation(
    prompt_data: dict,
    out_dir: Path,
    batch_size: int = 1,
    seed: int | None = None,
    generator_url: str = "https://perchance.org/ai-text-to-image-generator",
    headless: bool = True,
) -> list[Path]:
    """
    Submit prompt_data to perchance and download generated images.

    Args:
        prompt_data:   Dict from prompt_builder.build_prompt()
        out_dir:       Directory to save downloaded images
        batch_size:    Number of images to generate (1–15)
        seed:          Optional seed for reproducibility
        generator_url: Perchance generator URL
        headless:      Run browser headlessly (False for debugging)

    Returns:
        List of paths to saved images
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    positive = prompt_data["positive_prompt"]
    negative = prompt_data["negative_prompt"]
    style    = prompt_data.get("style_selector", "Professional Photo")

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_paths = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        print(f"Loading {generator_url}...")
        page.goto(generator_url, wait_until="networkidle", timeout=30000)
        # Give the page a moment to fully render
        page.wait_for_timeout(2000)

        # --- Fill positive prompt ---
        prompt_el = page.locator(SELECTORS["prompt_input"]).first
        prompt_el.wait_for(state="visible", timeout=10000)
        prompt_el.triple_click()
        prompt_el.fill(positive)
        print(f"  Prompt filled ({len(positive)} chars)")

        # --- Fill negative prompt (best-effort — not all generators have it) ---
        try:
            neg_el = page.locator(SELECTORS["negative_input"]).first
            neg_el.wait_for(state="visible", timeout=3000)
            neg_el.triple_click()
            neg_el.fill(negative)
            print(f"  Negative prompt filled ({len(negative)} chars)")
        except PWTimeout:
            print("  No negative prompt field found — skipping")

        # --- Set style dropdown ---
        try:
            style_el = page.locator(SELECTORS["style_select"]).first
            style_el.wait_for(state="visible", timeout=3000)
            style_el.select_option(label=style)
            print(f"  Style set to: {style}")
        except PWTimeout:
            print(f"  No style dropdown found — skipping (add '{style}' to your prompt manually)")
        except Exception as e:
            print(f"  Style selection failed ({e}) — skipping")

        # --- Set seed if provided ---
        if seed is not None:
            try:
                seed_el = page.locator(SELECTORS["seed_input"]).first
                seed_el.wait_for(state="visible", timeout=2000)
                seed_el.triple_click()
                seed_el.fill(str(seed))
                print(f"  Seed set to: {seed}")
            except PWTimeout:
                print(f"  No seed input found — seed {seed} not applied")

        # --- Set batch size ---
        batch_size = max(1, min(15, batch_size))
        if batch_size > 1:
            try:
                batch_el = page.locator(SELECTORS["batch_input"]).first
                batch_el.wait_for(state="visible", timeout=2000)
                batch_el.triple_click()
                batch_el.fill(str(batch_size))
                print(f"  Batch size set to: {batch_size}")
            except PWTimeout:
                print(f"  No batch input found — generating single image")

        # --- Click Generate ---
        gen_button = page.locator(SELECTORS["generate_button"]).first
        gen_button.wait_for(state="visible", timeout=5000)
        gen_button.click()
        print(f"  Generation started, waiting up to {GENERATION_TIMEOUT_S}s...")

        # --- Wait for image(s) to appear ---
        image_locator = page.locator(SELECTORS["result_images"])
        try:
            image_locator.first.wait_for(state="visible", timeout=GENERATION_TIMEOUT_S * 1000)
        except PWTimeout:
            print(f"  ERROR: No images appeared within {GENERATION_TIMEOUT_S}s")
            browser.close()
            return []

        # Give all batch images time to finish loading
        if batch_size > 1:
            page.wait_for_timeout(3000)

        # --- Download images ---
        images = image_locator.all()
        print(f"  {len(images)} image(s) found")

        for idx, img_el in enumerate(images):
            try:
                src = img_el.get_attribute("src") or ""
                if src.startswith("data:image"):
                    # Inline base64 — decode and save
                    import base64
                    header, b64 = src.split(",", 1)
                    ext = "jpg" if "jpeg" in header else "png"
                    fname = f"synthetic_{timestamp}_{idx:02d}.{ext}"
                    out_path = out_dir / fname
                    out_path.write_bytes(base64.b64decode(b64))
                    saved_paths.append(out_path)
                    print(f"  Saved: {fname}")
                elif src.startswith("blob:") or src.startswith("http"):
                    # Download via Playwright response interception or direct URL
                    fname = f"synthetic_{timestamp}_{idx:02d}.png"
                    out_path = out_dir / fname
                    response = context.request.get(src)
                    out_path.write_bytes(response.body())
                    saved_paths.append(out_path)
                    print(f"  Saved: {fname}")
                else:
                    print(f"  Image {idx}: unrecognised src format — skipping")
            except Exception as e:
                print(f"  Image {idx}: download failed ({e})")

        # Save prompt metadata alongside images
        meta_path = out_dir / f"synthetic_{timestamp}_prompt.json"
        meta_path.write_text(json.dumps({
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "generator_url":  generator_url,
            "prompt_data":    prompt_data,
            "seed":           seed,
            "batch_size":     batch_size,
            "saved_images":   [str(p) for p in saved_paths],
        }, indent=2))
        print(f"  Metadata saved: {meta_path.name}")

        browser.close()

    return saved_paths


def dump_selectors(url: str) -> None:
    """Print all textarea/select/button elements on the page for selector debugging."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed.")
        sys.exit(1)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)

        for tag in ["textarea", "select", "button", "input"]:
            els = page.locator(tag).all()
            print(f"\n--- {tag} elements ({len(els)}) ---")
            for el in els:
                attrs = {}
                for attr in ["id", "class", "placeholder", "name", "type"]:
                    v = el.get_attribute(attr)
                    if v:
                        attrs[attr] = v
                text = el.inner_text()[:60].strip() if tag == "button" else ""
                print(f"  {attrs}  text={repr(text)}")

        browser.close()


def main():
    parser = argparse.ArgumentParser(description="Perchance.org automation driver")
    parser.add_argument("prompt_json", nargs="?", type=Path, help="Path to prompt JSON from prompt_builder")
    parser.add_argument("--out",     type=Path,  default=REPO_ROOT / "synthetic", help="Output directory")
    parser.add_argument("--batch",   type=int,   default=1, help="Number of images to generate (1-15)")
    parser.add_argument("--seed",    type=int,   default=None, help="Seed for reproducibility")
    parser.add_argument("--url",     type=str,   default="https://perchance.org/ai-text-to-image-generator")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window (debugging)")
    parser.add_argument("--dump-selectors", action="store_true", help="Dump DOM elements and exit")
    args = parser.parse_args()

    if args.dump_selectors:
        dump_selectors(args.url)
        return

    if not args.prompt_json:
        parser.print_help()
        sys.exit(1)

    prompt_data = json.loads(args.prompt_json.read_text())
    saved = run_generation(
        prompt_data,
        out_dir=args.out,
        batch_size=args.batch,
        seed=args.seed,
        generator_url=args.url,
        headless=not args.no_headless,
    )
    print(f"\nDone. {len(saved)} image(s) saved to {args.out}")


if __name__ == "__main__":
    main()
