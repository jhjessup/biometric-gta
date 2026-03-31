"""
Autonomous 6-Hour Calibration Loop
===================================
Iterates through Perchance generator channels, GT artifacts, and prompt tuning
without human intervention. Runs until deadline or a perfect score is achieved.

Usage:
    python -m scripts.auto_loop [--hours 6] [--batch 3] [--target-score 11]

Loop logic:
  - For each channel: run calibration, track score, detect plateau
  - Plateau = no score improvement for PLATEAU_THRESHOLD consecutive runs
  - On plateau: move to next channel
  - When all channels plateau on current GT artifact: rotate to next GT artifact
  - Every FULL_BODY_EVERY face runs: run one full-body generation (extra data)
  - Use worker (OpenRouter) to analyze delta and suggest prompt adjustments
  - Commit results to git every GIT_COMMIT_EVERY runs
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.prompt_builder import build_prompt
from agents.geometry_analyzer import analyze_geometry
from agents.landmark_validator import validate_landmarks
from scripts.calibration_loop import run_calibration, _load_tuner, CALIBRATION_KEYS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOURS = 6
DEFAULT_BATCH = 3
DEFAULT_TARGET_SCORE = 11
PLATEAU_THRESHOLD = 3       # consecutive non-improving runs before channel switch
FULL_BODY_EVERY = 4         # face runs between each full-body run
GIT_COMMIT_EVERY = 5        # runs between git commits

# Known GT artifacts sorted by symmetry (lowest = most symmetric)
GT_ARTIFACTS = [
    "catalog/sessions/2026-03-29_shannon_001/artifacts/539144fb-7a3e-4392-b810-6de869005408.json",
    "catalog/sessions/2026-03-29_shannon_002/artifacts/cae3bd50-0000-0000-0000-000000000000.json",  # resolved below
    "catalog/sessions/2026-03-29_shannon_001/artifacts/91baaa03-0000-0000-0000-000000000000.json",  # resolved below
    "catalog/sessions/2026-03-29_shannon_001/artifacts/705bac62-0000-0000-0000-000000000000.json",  # resolved below
    "catalog/sessions/2026-03-29_shannon_001/artifacts/dcffbd31-0000-0000-0000-000000000000.json",  # resolved below
]

# Perchance generator channels to probe and iterate
# First entry is known-good baseline; rest are candidates to probe
CANDIDATE_CHANNELS = [
    "ai-text-to-image-generator",       # baseline: 10/11 @ guidance 7.0
    "ai-portrait-generator",            # worker-confirmed
    "realistic-portrait-generator",     # worker-confirmed
    "photorealistic-ai-generator",      # worker-confirmed
    "ai-realistic-image-generator",     # worker-confirmed
    "portrait-ai-generator",            # worker-confirmed
    "realistic-photo-generator",        # worker-confirmed
    "photo-realistic-portrait-ai",      # worker-confirmed
    "photorealistic-ai-image-generator",
    "realistic-ai-image-generator",
    "ai-photo-generator",
    "ai-art-generator",
]

GUIDANCE = 7.0
TUNER_PATH = REPO_ROOT / "scripts" / "calibration_tuner.json"
BEST_TUNE_PATH = REPO_ROOT / "synthetic" / "c6816c30" / "calibration.json"
LOG_PATH = REPO_ROOT / "synthetic" / "auto_loop_log.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_gt_artifacts() -> list[Path]:
    """Find actual artifact paths by prefix-matching artifact_id."""
    prefixes = ["539144fb", "cae3bd50", "91baaa03", "705bac62", "dcffbd31"]
    results = []
    all_artifacts = sorted(REPO_ROOT.glob("catalog/sessions/*/artifacts/*.json"))
    for prefix in prefixes:
        match = next((p for p in all_artifacts if p.stem.startswith(prefix)), None)
        if match:
            results.append(match)
        else:
            print(f"  [warn] GT artifact prefix {prefix} not found in catalog — skipping")
    return results


def _score_calibration(record: dict) -> int:
    """Count metrics with status 'ok' or 'ceiling'."""
    delta = record.get("delta", {})
    return sum(1 for v in delta.values() if v.get("status") in ("ok", "ceiling"))


def _probe_channel(channel_slug: str, gt_artifact: Path) -> bool:
    """
    Send a single test generation to a channel slug.
    Returns True if valid image returned, False if channel invalid/empty.
    """
    from scripts.perchance_http_client import run_generation
    import tempfile

    artifact = json.loads(gt_artifact.read_text())
    prompt_data = build_prompt(artifact)
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        url = f"https://perchance.org/{channel_slug}"
        saved = run_generation(
            prompt_data,
            out_dir=tmp_dir,
            batch_size=1,
            generator_url=url,
        )
        return len(saved) > 0
    except Exception as e:
        print(f"  [probe] {channel_slug}: error — {e}")
        return False


def _worker_analyze(record: dict) -> str:
    """Use OpenRouter worker to suggest prompt adjustments from delta."""
    failing = {
        k: v for k, v in record.get("delta", {}).items()
        if v.get("status") not in ("ok", "ceiling", "missing")
    }
    if not failing:
        return ""

    delta_lines = "\n".join(
        f"  {k}: GT={v['gt']:.3f} synthetic={v['synthetic']:.3f} delta={v['delta']:+.3f} ({v['status']})"
        for k, v in failing.items()
    )
    current_prompt = record.get("prompt_data", {}).get("positive_prompt", "")

    task = f"""You are a Stable Diffusion prompt engineer for a biometric calibration pipeline.
Current positive prompt: {current_prompt}

Failing metrics (synthetic vs ground truth):
{delta_lines}

Suggest ONLY weighted token additions or replacements in SD syntax like (token:weight).
Format your response as a JSON object with two keys:
  "append_positive": "additional tokens to append (comma-separated)"
  "replace_tokens": {{"old_token": "new_token"}} (for replacing existing tokens)

Keep it minimal — one or two targeted changes. Do not rewrite the whole prompt."""

    try:
        result = subprocess.run(
            ["python3", "/root/claude/worker.py",
             "--model", "qwen/qwen3.6-plus-preview:free",
             "--task", task,
             "--max-tokens", "256"],
            capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT)
        )
        output = result.stdout.strip()
        # Extract JSON from output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start >= 0 and end > start:
            return output[start:end]
    except Exception as e:
        print(f"  [worker] analysis failed: {e}")
    return ""


def _git_commit(run_count: int):
    try:
        subprocess.run(
            ["git", "add", "synthetic/", "progress.md"],
            cwd=str(REPO_ROOT), capture_output=True
        )
        msg = f"Auto-loop: {run_count} calibration runs completed\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(REPO_ROOT), capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(REPO_ROOT), capture_output=True
        )
        print(f"  [git] committed and pushed after {run_count} runs")
    except Exception as e:
        print(f"  [git] commit failed: {e}")


def _append_log(entry: dict):
    log = []
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    log.append(entry)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_auto_loop(
    max_hours: float = DEFAULT_HOURS,
    batch_size: int = DEFAULT_BATCH,
    target_score: int = DEFAULT_TARGET_SCORE,
):
    deadline = time.time() + max_hours * 3600
    print(f"\n{'='*60}")
    print(f"AUTO LOOP START — deadline in {max_hours}h")
    print(f"Target: {target_score}/11  |  Batch: {batch_size}  |  Guidance: {GUIDANCE}")
    print(f"{'='*60}\n")

    from scripts.perchance_http_client import run_generation

    # Resolve GT artifact paths
    gt_artifacts = _resolve_gt_artifacts()
    if not gt_artifacts:
        print("ERROR: No GT artifacts found. Exiting.")
        return

    print(f"GT artifacts loaded: {len(gt_artifacts)}")
    for p in gt_artifacts:
        print(f"  {p.stem[:8]}  {p.parent.parent.name}")

    # Probe channels
    print(f"\nProbing {len(CANDIDATE_CHANNELS)} channels...")
    valid_channels = []
    for slug in CANDIDATE_CHANNELS:
        if slug == "ai-text-to-image-generator":
            valid_channels.append(slug)
            print(f"  [known-good] {slug}")
            continue
        if time.time() >= deadline:
            break
        ok = _probe_channel(slug, gt_artifacts[0])
        status = "OK" if ok else "skip"
        print(f"  [{status}] {slug}")
        if ok:
            valid_channels.append(slug)
        time.sleep(15)  # rate limit between probes

    print(f"\nValid channels: {valid_channels}\n")

    tuner = _load_tuner(TUNER_PATH) if TUNER_PATH.exists() else None

    # Load best known tuning as starting point
    prior_tuning = None
    if BEST_TUNE_PATH.exists():
        try:
            prior_tuning = json.loads(BEST_TUNE_PATH.read_text()).get("prompt_overrides")
            print(f"Loaded prior tuning from c6816c30")
        except Exception:
            pass

    total_runs = 0
    best_ever_score = 0
    best_ever_run_id = None

    gt_idx = 0
    all_channels_plateau_count = 0  # how many consecutive GT rotations with no improvement

    while time.time() < deadline:
        gt_artifact = gt_artifacts[gt_idx % len(gt_artifacts)]
        print(f"\n{'─'*60}")
        print(f"GT artifact: {gt_artifact.stem[:8]}  ({gt_artifact.name})")
        print(f"{'─'*60}")

        channel_best_scores = {}
        all_plateaued = True

        for channel in valid_channels:
            if time.time() >= deadline:
                break

            print(f"\n  Channel: {channel}")
            channel_url = f"https://perchance.org/{channel}"
            consecutive_no_improve = 0
            channel_best = 0
            current_tuning = prior_tuning.copy() if prior_tuning else None

            while consecutive_no_improve < PLATEAU_THRESHOLD and time.time() < deadline:
                full_body = (total_runs % FULL_BODY_EVERY == FULL_BODY_EVERY - 1)
                mode_label = "full-body" if full_body else "face"
                print(f"\n    Run #{total_runs + 1} [{mode_label}]  channel={channel}  gt={gt_artifact.stem[:8]}")

                try:
                    record = run_calibration(
                        target_artifact_path=gt_artifact,
                        batch_size=batch_size,
                        prior_tuning=current_tuning,
                        generator_url=channel_url,
                        guidance_scale=GUIDANCE,
                        tuner=tuner,
                        generation_fn=run_generation,
                        full_body=full_body,
                    )
                except Exception as e:
                    print(f"    [error] run_calibration failed: {e}")
                    total_runs += 1
                    time.sleep(5)
                    continue

                score = _score_calibration(record)
                run_id = record.get("run_id", "?")
                print(f"    Score: {score}/11  run_id={run_id}")

                if score >= target_score:
                    print(f"\n{'*'*60}")
                    print(f"TARGET SCORE REACHED: {score}/11  run_id={run_id}")
                    print(f"Channel: {channel}  GT: {gt_artifact.stem[:8]}")
                    print(f"{'*'*60}\n")
                    _append_log({"event": "target_reached", "run_id": run_id, "score": score,
                                 "channel": channel, "gt": str(gt_artifact), "at": _now_iso()})
                    _git_commit(total_runs)
                    return

                if score > best_ever_score:
                    best_ever_score = score
                    best_ever_run_id = run_id
                    print(f"    *** New overall best: {score}/11 ***")

                if score > channel_best:
                    channel_best = score
                    consecutive_no_improve = 0
                    all_plateaued = False
                    # Use worker to analyze and improve prompt for next run
                    print("    Calling worker for prompt analysis...")
                    suggestion_json = _worker_analyze(record)
                    if suggestion_json:
                        try:
                            suggestion = json.loads(suggestion_json)
                            current_tuning = {
                                "append_positive": suggestion.get("append_positive", ""),
                                "replace_tokens": suggestion.get("replace_tokens", {}),
                            }
                            print(f"    Worker suggests: {suggestion_json[:120]}")
                        except Exception:
                            pass
                else:
                    consecutive_no_improve += 1
                    print(f"    No improvement ({consecutive_no_improve}/{PLATEAU_THRESHOLD})")

                channel_best_scores[channel] = channel_best

                _append_log({
                    "run": total_runs + 1,
                    "run_id": run_id,
                    "channel": channel,
                    "gt": gt_artifact.stem[:8],
                    "score": score,
                    "full_body": full_body,
                    "best_ever": best_ever_score,
                    "at": _now_iso(),
                })

                total_runs += 1

                if total_runs % GIT_COMMIT_EVERY == 0:
                    _git_commit(total_runs)

        # After iterating all channels on this GT artifact
        if all_plateaued:
            all_channels_plateau_count += 1
            print(f"\n  All channels plateaued on GT {gt_artifact.stem[:8]}. "
                  f"Best scores: {channel_best_scores}. Rotating GT artifact...")
            gt_idx += 1
            prior_tuning = None  # reset tuning for new GT
            if gt_idx >= len(gt_artifacts):
                gt_idx = 0
                print("  All GT artifacts cycled. Restarting from first.")
        else:
            all_channels_plateau_count = 0

    # Deadline reached
    print(f"\n{'='*60}")
    print(f"AUTO LOOP COMPLETE — {total_runs} total runs")
    print(f"Best score: {best_ever_score}/11  run_id={best_ever_run_id}")
    print(f"{'='*60}\n")
    _git_commit(total_runs)

    # Write summary to progress.md append
    summary = f"""
---

### Auto-Loop Summary ({_now_iso()[:10]})

- Total runs: {total_runs}
- Best score: {best_ever_score}/11  (run `{best_ever_run_id}`)
- Channels tested: {', '.join(valid_channels)}
- GT artifacts rotated: {gt_idx} times
- Duration: {max_hours}h
"""
    progress = REPO_ROOT / "progress.md"
    progress.write_text(progress.read_text() + summary)
    _git_commit(total_runs)


def main():
    parser = argparse.ArgumentParser(description="6-hour autonomous calibration loop")
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--target-score", type=int, default=DEFAULT_TARGET_SCORE)
    args = parser.parse_args()
    run_auto_loop(max_hours=args.hours, batch_size=args.batch, target_score=args.target_score)


if __name__ == "__main__":
    main()
