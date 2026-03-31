"""
Perchance.org Direct HTTP Client
Replaces the Playwright-based perchance_driver.py with a direct HTTP client.
Uses curl_cffi to impersonate Chrome's TLS fingerprint (bypasses Cloudflare).

API flow (reverse-engineered):
  1. GET  https://perchance.org/api/getAccessCodeForAdPoweredStuff
         → 64-char access code (ad-verification)
  2. Obtain a Cloudflare Turnstile token for site key 0x4AAAAAAAi3LdM-EVMMMFCv
         → Either via a solver service (see TURNSTILE_SOLVER below) or manually
  3. GET  https://image-generation.perchance.org/api/verifyUser?token=<turnstile>
         → parse 64-char userKey from JSON response
  4. POST https://image-generation.perchance.org/api/generate?userKey=...
         → returns {imageId, fileExtension, seed, ...}
  5. GET  https://image-generation.perchance.org/api/downloadTemporaryImage?imageId=...
         → binary image bytes

Turnstile solver options (set TURNSTILE_API_KEY in .env):
  - CapSolver  https://capsolver.com   (set TURNSTILE_SOLVER=capsolver)
  - 2captcha   https://2captcha.com    (set TURNSTILE_SOLVER=2captcha)
  - Manual     provide token via --turnstile-token CLI arg or TURNSTILE_TOKEN env var

Usage (CLI):
    python -m scripts.perchance_http_client <prompt_json> [--out DIR] [--batch N] [--seed SEED]
    python -m scripts.perchance_http_client <prompt_json> --turnstile-token <token>

Usage (library — drop-in for perchance_driver.run_generation):
    from scripts.perchance_http_client import run_generation
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

API_BASE          = "https://image-generation.perchance.org/api"
PERCHANCE_ORIGIN  = "https://perchance.org"
GENERATOR_PAGE    = "https://perchance.org/ai-text-to-image-generator"
TURNSTILE_SITEKEY = "0x4AAAAAAAi3LdM-EVMMMFCv"
DEFAULT_URL       = GENERATOR_PAGE

INTER_REQUEST_DELAY_S = 13   # ~5 req/min rate limit

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": GENERATOR_PAGE,
    "Origin":  PERCHANCE_ORIGIN,
}


# ---------------------------------------------------------------------------
# Turnstile token acquisition
# ---------------------------------------------------------------------------

def _solve_turnstile_capsolver(api_key: str) -> str:
    """Solve Turnstile via CapSolver API. Returns token string."""
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        raise RuntimeError("curl_cffi not installed: pip install curl_cffi")

    # Create task
    task_resp = cffi.post(
        "https://api.capsolver.com/createTask",
        json={
            "clientKey": api_key,
            "task": {
                "type": "AntiTurnstileTaskProxyLess",
                "websiteURL": GENERATOR_PAGE,
                "websiteKey": TURNSTILE_SITEKEY,
            },
        },
        impersonate="chrome120",
        timeout=15,
    )
    task_data = task_resp.json()
    if task_data.get("errorId", 0) != 0:
        raise RuntimeError(f"CapSolver createTask error: {task_data.get('errorDescription')}")
    task_id = task_data["taskId"]

    # Poll for result
    for _ in range(30):
        time.sleep(3)
        res = cffi.post(
            "https://api.capsolver.com/getTaskResult",
            json={"clientKey": api_key, "taskId": task_id},
            impersonate="chrome120",
            timeout=10,
        )
        data = res.json()
        if data.get("status") == "ready":
            return data["solution"]["token"]
        if data.get("status") == "failed":
            raise RuntimeError(f"CapSolver task failed: {data}")
    raise RuntimeError("CapSolver: timed out waiting for Turnstile token")


def _solve_turnstile_2captcha(api_key: str) -> str:
    """Solve Turnstile via 2captcha API. Returns token string."""
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        raise RuntimeError("curl_cffi not installed: pip install curl_cffi")

    r = cffi.get(
        "https://2captcha.com/in.php",
        params={
            "key": api_key,
            "method": "turnstile",
            "sitekey": TURNSTILE_SITEKEY,
            "pageurl": GENERATOR_PAGE,
            "json": "1",
        },
        impersonate="chrome120",
        timeout=15,
    )
    data = r.json()
    if data.get("status") != 1:
        raise RuntimeError(f"2captcha submit error: {data}")
    captcha_id = data["request"]

    for _ in range(30):
        time.sleep(5)
        res = cffi.get(
            "https://2captcha.com/res.php",
            params={"key": api_key, "action": "get", "id": captcha_id, "json": "1"},
            impersonate="chrome120",
            timeout=10,
        )
        d = res.json()
        if d.get("status") == 1:
            return d["request"]
        if d.get("request") not in ("CAPCHA_NOT_READY", "CAPTCHA_NOT_READY"):
            raise RuntimeError(f"2captcha error: {d}")
    raise RuntimeError("2captcha: timed out waiting for Turnstile token")


def get_turnstile_token(manual_token: str | None = None) -> str:
    """
    Obtain a Cloudflare Turnstile token by one of three methods (in priority order):
      1. manual_token argument / TURNSTILE_TOKEN env var
      2. TURNSTILE_API_KEY + TURNSTILE_SOLVER env vars (capsolver or 2captcha)
      3. Raise RuntimeError with instructions
    """
    token = manual_token or os.environ.get("TURNSTILE_TOKEN", "").strip()
    if token:
        return token

    api_key = os.environ.get("TURNSTILE_API_KEY", "").strip()
    solver  = os.environ.get("TURNSTILE_SOLVER", "capsolver").lower()
    if api_key:
        print(f"  Solving Turnstile via {solver}...")
        if solver == "capsolver":
            return _solve_turnstile_capsolver(api_key)
        elif solver == "2captcha":
            return _solve_turnstile_2captcha(api_key)
        else:
            raise RuntimeError(f"Unknown TURNSTILE_SOLVER: {solver!r} (use 'capsolver' or '2captcha')")

    raise RuntimeError(
        "Cloudflare Turnstile token required but none provided.\n"
        "Options:\n"
        "  1. Set TURNSTILE_TOKEN=<token> in .env (get one manually from browser DevTools)\n"
        "  2. Set TURNSTILE_API_KEY=<key> and TURNSTILE_SOLVER=capsolver in .env\n"
        "     (sign up at https://capsolver.com — ~$1-2 per 1000 tokens)\n"
        "  3. Set TURNSTILE_API_KEY=<key> and TURNSTILE_SOLVER=2captcha in .env\n"
        "     (sign up at https://2captcha.com)\n"
        "  4. Run on a local machine with: python -m scripts.perchance_driver"
    )


# ---------------------------------------------------------------------------
# Core API calls
# ---------------------------------------------------------------------------

def _get_user_key(sess, turnstile_token: str) -> str:
    """
    Fetch a fresh userKey from the verifyUser endpoint.
    Returns the 64-character hex key string.
    """
    params = {
        "thread":       "0",
        "token":        turnstile_token,
        "__cacheBust":  str(random.random()),
    }
    resp = sess.get(f"{API_BASE}/verifyUser", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") == "failed_verification":
        reason = data.get("reason", "unknown")
        raise RuntimeError(f"perchance verifyUser failed: {reason}")

    match = re.search(r'"userKey"\s*:\s*"([0-9a-f]{64})"', resp.text)
    if not match:
        raise RuntimeError(
            f"perchance verifyUser: could not parse userKey. Response: {resp.text[:300]!r}"
        )
    return match.group(1)


def _generate_one(
    sess,
    user_key: str,
    prompt: str,
    negative_prompt: str,
    channel: str,
    resolution: str = "512x768",
    guidance_scale: float = 7.0,
    seed: int = -1,
) -> dict:
    """Submit one generation request. Returns the API response dict."""
    params = {
        "userKey":     user_key,
        "requestId":   f"aiImageCompletion{random.randint(0, 2**30)}",
        "__cacheBust": str(random.random()),
    }
    body = {
        "generatorName":  "ai-image-generator",
        "channel":        channel,
        "subChannel":     "public",
        "prompt":         prompt,
        "negativePrompt": negative_prompt,
        "seed":           seed,
        "resolution":     resolution,
        "guidanceScale":  guidance_scale,
    }
    resp = sess.post(
        f"{API_BASE}/generate",
        params=params,
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    if "imageId" not in data:
        raise RuntimeError(f"perchance generate: unexpected response: {data!r}")
    return data


def _download_image(sess, image_id: str) -> bytes:
    """Download a generated image by its imageId. Returns raw bytes."""
    resp = sess.get(
        f"{API_BASE}/downloadTemporaryImage",
        params={"imageId": image_id},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def _channel_from_url(generator_url: str | None) -> str:
    return (generator_url or DEFAULT_URL).rstrip("/").split("/")[-1]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_generation(
    prompt_data: dict,
    out_dir: Path,
    batch_size: int = 1,
    seed: int | None = None,
    generator_url: str = DEFAULT_URL,
    headless: bool = True,          # ignored — no browser; kept for API compat
    resolution: str = "512x768",
    guidance_scale: float = 7.0,
    turnstile_token: str | None = None,
    user_key: str | None = None,
    **kwargs,                        # absorb backend-specific params (e.g. reference_image_path, model)
) -> list[Path]:
    """
    Generate images via the perchance HTTP API. Drop-in for perchance_driver.run_generation().

    Args:
        prompt_data:      Dict from agents.prompt_builder.build_prompt()
        out_dir:          Directory to save downloaded images
        batch_size:       Number of images to generate (1–15)
        seed:             Optional fixed seed; None = random per image
        generator_url:    Perchance generator URL (used to derive channel name)
        headless:         Ignored — kept for drop-in compatibility
        resolution:       "512x768" (portrait), "768x768" (square), "768x512" (landscape)
        guidance_scale:   1–30 (default 7.0)
        turnstile_token:  Cloudflare Turnstile token. If None, reads TURNSTILE_TOKEN env var
                          or calls a configured solver service.
        user_key:         64-char perchance userKey (skips verifyUser entirely).
                          Copy from browser DevTools → verifyUser response JSON.
                          Set PERCHANCE_USER_KEY in .env as an alternative.

    Returns:
        List of Paths to saved images.
    """
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        print("curl_cffi not installed. Run: pip install curl_cffi")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    positive = prompt_data["positive_prompt"]
    negative = prompt_data.get("negative_prompt", "")
    style    = prompt_data.get("style_selector", "")
    channel  = _channel_from_url(generator_url)

    if style:
        positive = f"{positive}, {style} style"

    batch_size = max(1, min(15, batch_size))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    saved_paths: list[Path] = []

    sess = cffi.Session()
    sess.impersonate = "chrome120"
    sess.headers.update(_BROWSER_HEADERS)

    # Resolve userKey — direct supply skips Turnstile entirely
    resolved_key = user_key or os.environ.get("PERCHANCE_USER_KEY", "").strip()
    if resolved_key:
        print(f"Using supplied userKey: {resolved_key[:8]}...{resolved_key[-8:]}")
    else:
        # Fall back to Turnstile → verifyUser flow
        print("Obtaining Turnstile token...")
        try:
            token = get_turnstile_token(turnstile_token)
            print(f"  token: {token[:12]}...")
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            return []

        print("Fetching userKey...")
        try:
            resolved_key = _get_user_key(sess, token)
            print(f"  userKey: {resolved_key[:8]}...{resolved_key[-8:]}")
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            return []

    for idx in range(batch_size):
        img_seed = seed if seed is not None else -1
        print(f"\nGenerating image {idx + 1}/{batch_size} (seed={img_seed})...")

        try:
            gen_data    = _generate_one(sess, resolved_key, positive, negative, channel,
                                        resolution=resolution, guidance_scale=guidance_scale,
                                        seed=img_seed)
            image_id    = gen_data["imageId"]
            ext         = gen_data.get("fileExtension", "png")
            print(f"  imageId: {image_id}  ext={ext}  seed={gen_data.get('seed')}")

            image_bytes = _download_image(sess, image_id)
            fname       = f"synthetic_{timestamp}_{idx:02d}.{ext}"
            out_path    = out_dir / fname
            out_path.write_bytes(image_bytes)
            saved_paths.append(out_path)
            print(f"  Saved: {fname} ({len(image_bytes):,} bytes)")

        except Exception as e:
            print(f"  ERROR generating image {idx + 1}: {e}")

        if idx < batch_size - 1:
            print(f"  Waiting {INTER_REQUEST_DELAY_S}s (rate limit)...")
            time.sleep(INTER_REQUEST_DELAY_S)

    meta_path = out_dir / f"synthetic_{timestamp}_prompt.json"
    meta_path.write_text(json.dumps({
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "generator_url":  generator_url,
        "channel":        channel,
        "prompt_data":    prompt_data,
        "positive_sent":  positive,
        "negative_sent":  negative,
        "seed":           seed,
        "resolution":     resolution,
        "guidance_scale": guidance_scale,
        "batch_size":     batch_size,
        "saved_images":   [str(p) for p in saved_paths],
    }, indent=2))
    print(f"\nMetadata saved: {meta_path.name}")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Perchance HTTP client — server-safe, no browser required"
    )
    parser.add_argument("prompt_json", nargs="?", type=Path)
    parser.add_argument("--out",              type=Path,  default=REPO_ROOT / "synthetic")
    parser.add_argument("--batch",            type=int,   default=1)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--url",              type=str,   default=DEFAULT_URL)
    parser.add_argument("--resolution",       type=str,   default="512x768",
                        help="512x768 (portrait) | 768x768 (square) | 768x512 (landscape)")
    parser.add_argument("--guidance",         type=float, default=7.0)
    parser.add_argument("--turnstile-token",  type=str,   default=None,
                        help="Pre-solved Cloudflare Turnstile token (or set TURNSTILE_TOKEN in .env)")
    parser.add_argument("--user-key",         type=str,   default=None,
                        help="64-char perchance userKey from browser DevTools (skips Turnstile). Or set PERCHANCE_USER_KEY in .env")
    args = parser.parse_args()

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
        resolution=args.resolution,
        guidance_scale=args.guidance,
        turnstile_token=args.turnstile_token,
        user_key=args.user_key,
    )
    print(f"\nDone. {len(saved)} image(s) saved to {args.out}")


if __name__ == "__main__":
    main()
