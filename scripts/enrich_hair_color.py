"""
Backfill hair color analysis into all existing artifacts.

Adds artifact["enrichment"]["hair_analysis"] using agents/hair_analyzer.py.
Safe to re-run — skips artifacts that already have hair_analysis.

Usage:
    python -m scripts.enrich_hair_color [--force]

    --force   Re-analyze even if hair_analysis already present
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.hair_analyzer import analyze_hair_color


def _find_stripped_image(artifact: dict, artifact_path: Path) -> Path | None:
    """Locate the stripped source image for an artifact."""
    src_file = artifact.get("metadata", {}).get("source_filename", "")
    stem     = Path(src_file).stem
    session_dir = artifact_path.parent.parent   # artifacts/ → session root

    # Common locations
    candidates = [
        session_dir / "stripped" / f"{stem}_stripped.png",
        session_dir / "stripped" / f"{stem}.png",
        *session_dir.glob(f"**/{stem}_stripped.png"),
        *session_dir.glob(f"**/{stem}*.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run(force: bool = False) -> None:
    artifact_paths = sorted(REPO_ROOT.glob("catalog/sessions/*/artifacts/*.json"))
    print(f"Found {len(artifact_paths)} artifacts")

    skipped = done = failed = 0

    for art_path in artifact_paths:
        artifact = json.loads(art_path.read_text())
        src_file = artifact.get("metadata", {}).get("source_filename", art_path.name)

        # Skip if already done and not forcing
        if not force and artifact.get("enrichment", {}).get("hair_analysis"):
            skipped += 1
            continue

        stripped = _find_stripped_image(artifact, art_path)
        if stripped is None:
            print(f"  SKIP {src_file} — no stripped image found")
            skipped += 1
            continue

        try:
            result = analyze_hair_color(stripped, artifact)
            if "enrichment" not in artifact:
                artifact["enrichment"] = {}
            artifact["enrichment"]["hair_analysis"] = result
            art_path.write_text(json.dumps(artifact, indent=2))
            conf_str = f"conf={result['confidence']:.2f}"
            print(f"  OK   {src_file:30s}  {result['shade_descriptor']:25s}  {conf_str}")
            done += 1
        except Exception as e:
            print(f"  FAIL {src_file} — {e}")
            failed += 1

    print(f"\nDone: {done}  Skipped: {skipped}  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Backfill hair color analysis into all artifacts")
    parser.add_argument("--force", action="store_true", help="Re-analyze even if already present")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
