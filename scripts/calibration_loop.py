"""
Calibration Loop
Orchestrates: build prompt → generate via perchance → measure synthetic output
→ compute geometry delta vs. ground truth → emit tuning recommendations.

Each iteration writes a calibration record to synthetic/<run_id>/calibration.json.
Tuning recommendations are printed and can be fed back into prompt_builder manually
or via the --apply-tuning flag on the next run.

Usage:
    # Run one calibration iteration (generates + measures + reports delta)
    python -m scripts.calibration_loop <artifact.json> [--batch N] [--seed SEED] [--run-id ID]

    # Apply tuning from a prior run and re-generate
    python -m scripts.calibration_loop <artifact.json> --tune <calibration.json>

    # Dry-run: just build and print the prompt without generating
    python -m scripts.calibration_loop <artifact.json> --dry-run
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.prompt_builder   import build_prompt
from agents.geometry_analyzer import analyze_geometry
from agents.landmark_validator import validate_landmarks

# Measurements to compare in the calibration delta.
# These are the most prompt-responsive (directly described by text tokens).
CALIBRATION_KEYS = [
    "facial_index",
    "nasal_index",
    "jaw_to_cheek_ratio",
    "canthal_tilt_left_deg",
    "canthal_tilt_right_deg",
    "eye_left_aspect_ratio",
    "eye_right_aspect_ratio",
    "symmetry_index",
    "facial_third_lower",
    "mouth_to_iod_ratio",
    "outer_canthal_dist_norm",
]

# Acceptable delta thresholds per measurement before a tuning suggestion fires.
# Values are in the same units as the measurement (mixed: some % indexes, some ratios, some degrees).
DELTA_THRESHOLDS = {
    "facial_index":           8.0,   # index points
    "nasal_index":           12.0,
    "jaw_to_cheek_ratio":     0.06,
    "canthal_tilt_left_deg":  4.0,   # degrees
    "canthal_tilt_right_deg": 4.0,
    "eye_left_aspect_ratio":  0.04,
    "eye_right_aspect_ratio": 0.04,
    "symmetry_index":         0.10,
    "facial_third_lower":     0.04,
    "mouth_to_iod_ratio":     0.20,
    "outer_canthal_dist_norm":0.15,
}

# Tuning suggestion templates.
# Each entry maps a (measurement, direction) pair to a prompt adjustment suggestion.
# direction: "increase" means synthetic value is too low vs. ground truth.
TUNING_SUGGESTIONS = {
    ("facial_index", "increase"):    "Add (long face:1.1) or increase to (long face:1.2)",
    ("facial_index", "decrease"):    "Add (wide face:1.1) or increase to (wide face:1.2)",
    ("nasal_index",  "increase"):    "Add (broad nose:1.1) or increase weight",
    ("nasal_index",  "decrease"):    "Add (narrow nose:1.1) or increase weight",
    ("jaw_to_cheek_ratio","increase"):"Add (strong jawline:1.1) — wider jaw relative to cheeks",
    ("jaw_to_cheek_ratio","decrease"):"Add (defined cheekbones:1.1) — narrower jaw",
    ("canthal_tilt_left_deg","increase"):  "Add (upturned eyes:1.1)",
    ("canthal_tilt_left_deg","decrease"):  "Add (downturned eyes:1.1)",
    ("canthal_tilt_right_deg","increase"): "Add (upturned eyes:1.1)",
    ("canthal_tilt_right_deg","decrease"): "Add (downturned eyes:1.1)",
    ("eye_left_aspect_ratio","increase"):  "Add (large open eyes:1.1) or (wide eyes:1.1)",
    ("eye_left_aspect_ratio","decrease"):  "Add (narrow hooded eyes:1.1) or (almond eyes:1.1)",
    ("eye_right_aspect_ratio","increase"): "Add (large open eyes:1.1)",
    ("eye_right_aspect_ratio","decrease"): "Add (narrow hooded eyes:1.1)",
    ("symmetry_index","increase"):   "Asymmetry too low in synthetic — add (natural facial asymmetry:0.9) to [de-emphasize] symmetry",
    ("symmetry_index","decrease"):   "Asymmetry too high — remove any asymmetry tokens; add (perfectly symmetrical face:1.1)",
    ("facial_third_lower","increase"):"Add (long chin:1.1) or (prominent chin:1.1)",
    ("facial_third_lower","decrease"):"Add (short chin:1.1) or (receding chin:1.1)",
    ("mouth_to_iod_ratio","increase"):"Add (wide mouth:1.1) or (full lips:1.1)",
    ("mouth_to_iod_ratio","decrease"):"Add (small mouth:1.1) or (thin lips:1.1)",
    ("outer_canthal_dist_norm","increase"):"Add (wide-set eyes:1.1)",
    ("outer_canthal_dist_norm","decrease"):"Add (close-set eyes:1.1)",
}


def _load_tuner(tuner_path: Path | None) -> dict:
    """
    Load a synthesizer tuner config from JSON.

    Tuner schema:
      delta_thresholds  — per-key threshold overrides (float)
      ceiling_metrics   — list of keys treated as model ceiling (relax suggestions)
      tuning_suggestions — per-"key/direction" suggestion overrides (string)

    Returns an empty dict if no tuner is provided.
    """
    if tuner_path is None:
        return {}
    raw = json.loads(tuner_path.read_text())
    # Normalise tuning_suggestions keys from "key/direction" → (key, direction)
    raw_suggestions = raw.get("tuning_suggestions", {})
    raw["_tuning_suggestions_parsed"] = {
        tuple(k.split("/", 1)): v for k, v in raw_suggestions.items()
    }
    return raw


def _compute_delta(ground_truth: dict, synthetic: dict, tuner: dict | None = None) -> dict:
    """
    Compare geometry measurement dicts. Returns per-key delta analysis.

    tuner — optional synthesizer tuner config (from _load_tuner); overrides
             thresholds and suggestions for ceiling metrics.
    """
    tuner = tuner or {}
    threshold_overrides  = tuner.get("delta_thresholds", {})
    ceiling_metrics      = set(tuner.get("ceiling_metrics", []))
    suggestion_overrides = tuner.get("_tuning_suggestions_parsed", {})

    results = {}
    for key in CALIBRATION_KEYS:
        gt_val  = ground_truth.get(key)
        syn_val = synthetic.get(key)
        if gt_val is None or syn_val is None:
            results[key] = {"gt": gt_val, "synthetic": syn_val, "delta": None, "status": "missing"}
            continue
        delta     = syn_val - gt_val          # positive = synthetic > ground truth
        threshold = threshold_overrides.get(key, DELTA_THRESHOLDS.get(key, 0.1))
        is_ceiling = key in ceiling_metrics
        if abs(delta) <= threshold:
            status = "ok"
        elif is_ceiling:
            status = "ceiling"   # model cannot reach GT; no tuning suggestion will help
        elif delta > 0:
            status = "synthetic_high"
        else:
            status = "synthetic_low"
        direction = "decrease" if delta > 0 else "increase"
        if status in ("synthetic_high", "synthetic_low"):
            suggestion = suggestion_overrides.get((key, direction), TUNING_SUGGESTIONS.get((key, direction)))
        elif status == "ceiling":
            # Still surface the best known alternate token strategy
            suggestion = suggestion_overrides.get((key, direction), TUNING_SUGGESTIONS.get((key, direction)))
        else:
            suggestion = None
        results[key] = {
            "gt":        round(float(gt_val),  4),
            "synthetic": round(float(syn_val), 4),
            "delta":     round(float(delta),   4),
            "threshold": threshold,
            "status":    status,
            "suggestion": suggestion,
        }
    return results


def _measure_image(image_path: Path, stripped_dir: Path) -> dict | None:
    """
    Run the face landmark + geometry pipeline on a synthetic image.
    Returns geometry measurements dict or None.
    """
    import hashlib
    from agents.exif_stripper import strip_exif_to_file

    stripped_path = stripped_dir / f"{image_path.stem}_stripped.png"
    sha256 = strip_exif_to_file(image_path, stripped_path)

    artifact = validate_landmarks(stripped_path, sha256)
    if not artifact:
        return None

    artifact["geometry"] = analyze_geometry(artifact)
    return artifact["geometry"]["measurements"]


def run_calibration(
    target_artifact_path: Path,
    run_id: str | None = None,
    batch_size: int = 3,
    seed: int | None = None,
    dry_run: bool = False,
    prior_tuning: dict | None = None,
    generator_url: str | None = None,
    guidance_scale: float = 7.0,
    tuner: dict | None = None,
    reference_image_path: Path | None = None,
    ip_adapter_scale: float = 0.8,
    model: str = "qwen/qwen3.6-plus-preview",
    resolution: str = "512x768",
    generation_fn=None,
    full_body: bool = False,
) -> dict:
    """
    Execute one calibration iteration.

    Returns the calibration record dict.
    """
    run_id    = run_id or uuid.uuid4().hex[:8]
    out_dir   = REPO_ROOT / "synthetic" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ground truth artifact ---
    target = json.loads(target_artifact_path.read_text())
    gt_measurements = target["geometry"]["measurements"]

    print(f"\n{'='*60}")
    print(f"Calibration run: {run_id}")
    print(f"Target:          {target.get('metadata',{}).get('source_filename','?')}")
    print(f"Artifact ID:     {target['artifact_id']}")

    # --- Build prompt ---
    prompt_data = build_prompt(target, full_body=full_body)

    # Apply any tuning overrides from a prior run
    if prior_tuning:
        overrides = prior_tuning
        if overrides.get("append_positive"):
            prompt_data["positive_prompt"] += ", " + overrides["append_positive"]
        if overrides.get("replace_tokens"):
            for old, new in overrides["replace_tokens"].items():
                prompt_data["positive_prompt"] = prompt_data["positive_prompt"].replace(old, new)
        print(f"Applied {len(overrides)} tuning override(s) from prior run")

    print(f"\n--- POSITIVE PROMPT ---")
    print(prompt_data["positive_prompt"])
    print(f"\n--- NEGATIVE PROMPT ---")
    print(prompt_data["negative_prompt"])
    print(f"\n--- STYLE: {prompt_data['style_selector']} ---")

    # Save prompt to run dir
    prompt_path = out_dir / "prompt.json"
    prompt_path.write_text(json.dumps(prompt_data, indent=2))

    if dry_run:
        print("\n[dry-run] Skipping generation and measurement.")
        return {"run_id": run_id, "dry_run": True, "prompt_data": prompt_data}

    # --- Generate via backend ---
    if generation_fn is None:
        raise RuntimeError("generation_fn not provided to run_calibration(). Pass via main() backend selection.")
    print(f"\nGenerating {batch_size} image(s)...")
    saved_images = generation_fn(
        prompt_data,
        out_dir=out_dir,
        batch_size=batch_size,
        seed=seed,
        generator_url=generator_url,
        headless=True,
        guidance_scale=guidance_scale,
        reference_image_path=reference_image_path,
        ip_adapter_scale=ip_adapter_scale,
        model=model,
        resolution=resolution,
    )

    if not saved_images:
        print("No images generated — calibration cannot proceed.")
        return {"run_id": run_id, "error": "no_images_generated"}

    # --- Measure synthetic outputs ---
    stripped_dir = out_dir / "stripped"
    stripped_dir.mkdir(exist_ok=True)

    synthetic_measurements = []
    for img_path in saved_images:
        print(f"Measuring {img_path.name}...")
        try:
            meas = _measure_image(img_path, stripped_dir)
            if meas:
                synthetic_measurements.append(meas)
                print(f"  OK — facial_index={meas.get('facial_index')}  symmetry={meas.get('symmetry_index')}")
            else:
                print(f"  No face detected in synthetic image — skipping")
        except Exception as e:
            print(f"  Measurement failed: {e}")

    if not synthetic_measurements:
        print("No synthetic images could be measured.")
        return {"run_id": run_id, "error": "no_synthetic_measurements"}

    # Aggregate synthetic measurements (mean across batch)
    agg = {}
    for key in CALIBRATION_KEYS:
        vals = [m[key] for m in synthetic_measurements if m.get(key) is not None]
        agg[key] = float(np.mean(vals)) if vals else None

    # --- Compute delta ---
    delta = _compute_delta(gt_measurements, agg, tuner=tuner)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"CALIBRATION DELTA  (n={len(synthetic_measurements)} synthetic images)")
    print(f"{'Measurement':35s}  {'GT':>8}  {'Syn':>8}  {'Delta':>8}  Status")
    print("-" * 80)
    suggestions = []
    for key, r in delta.items():
        if r["delta"] is None:
            continue
        status_marker = "✓" if r["status"] in ("ok", "ceiling") else "✗"
        ceiling_tag = " [ceiling]" if r["status"] == "ceiling" else ""
        print(f"{status_marker} {key:33s}  {r['gt']:8.3f}  {r['synthetic']:8.3f}  {r['delta']:+8.3f}  {r['status']}{ceiling_tag}")
        if r.get("suggestion") and r["status"] not in ("ok",):
            suggestions.append(f"  [{key}] {r['suggestion']}")

    if suggestions:
        print(f"\nTUNING SUGGESTIONS:")
        for s in suggestions:
            print(s)
    else:
        print("\nAll measurements within threshold — prompt is well-calibrated.")

    # --- Save calibration record ---
    record = {
        "run_id":                run_id,
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "target_artifact_id":    target["artifact_id"],
        "target_source_file":    target.get("metadata", {}).get("source_filename"),
        "prompt_data":           prompt_data,
        "n_synthetic_images":    len(saved_images),
        "n_measured":            len(synthetic_measurements),
        "ground_truth":          {k: gt_measurements.get(k) for k in CALIBRATION_KEYS},
        "synthetic_mean":        agg,
        "delta":                 delta,
        "tuning_suggestions":    suggestions,
        "synthesizer_tuner": {
            "ceiling_metrics":     list(tuner.get("ceiling_metrics", [])) if tuner else [],
            "threshold_overrides": tuner.get("delta_thresholds", {}) if tuner else {},
        },
        "prompt_overrides": {
            "append_positive": "",
            "replace_tokens":  {},
            "_instructions": (
                "To apply tuning: populate append_positive with additional weighted tokens "
                "and/or replace_tokens with {old_token: new_token} pairs. "
                "Pass this file to the next run via --tune."
            ),
        },
    }

    record_path = out_dir / "calibration.json"
    record_path.write_text(json.dumps(record, indent=2))
    print(f"\nCalibration record saved: {record_path}")

    return record


def main():
    parser = argparse.ArgumentParser(description="GTA calibration loop for perchance.org")
    parser.add_argument("artifact", type=Path, help="Target GTA artifact JSON")
    parser.add_argument("--batch",    type=int,   default=3,   help="Images to generate per run")
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--run-id",   type=str,   default=None)
    parser.add_argument("--url",      type=str,   default=None,
                        help="InstantID server URL (overrides INSTANTID_SERVER_URL env var)")
    parser.add_argument("--dry-run",  action="store_true", help="Build prompt only, no generation")
    parser.add_argument("--tune",     type=Path,  default=None,
                        help="Path to a prior calibration.json with prompt_overrides populated")
    parser.add_argument("--guidance", type=float, default=7.0,
                        help="Guidance scale 1–30 (default 7.0; try 9.5 for tighter prompt compliance)")
    parser.add_argument("--tuner",     type=Path,  default=None,
                        help="Path to synthesizer tuner JSON (overrides thresholds/suggestions for ceiling metrics)")
    parser.add_argument("--reference", type=Path,  default=None,
                        help="EXIF-stripped source image to use as InstantID face reference. "
                             "If omitted, uses agents.reference_selector to auto-pick from the target session.")
    parser.add_argument("--ip-scale",  type=float, default=0.8,
                        help="InstantID identity strength 0.0–1.0 (default 0.8). "
                             "Higher = stronger identity lock, less generative variation.")
    parser.add_argument("--backend", choices=["openrouter", "instantid", "perchance"],
                        default="openrouter",
                        help="Generation backend (default: openrouter)")
    parser.add_argument("--model", default="qwen/qwen3.6-plus-preview",
                        help="Model ID for OpenRouter backend (default: qwen/qwen3.6-plus-preview)")
    parser.add_argument("--resolution", type=str, default="512x768",
                        help="Image resolution: 512x768 (portrait), 768x768 (square), 768x512 (landscape). "
                             "Passed to perchance backend only (default: 512x768)")
    args = parser.parse_args()

    if args.backend == "openrouter":
        from scripts.openrouter_client import run_generation
    elif args.backend == "instantid":
        from scripts.instantid_client import run_generation
    else:
        from scripts.perchance_http_client import run_generation

    prior_tuning = None
    if args.tune:
        prior_tuning = json.loads(args.tune.read_text()).get("prompt_overrides")

    tuner = _load_tuner(args.tuner) if args.tuner else None

    # Auto-select reference image if not specified
    reference_image_path = args.reference
    if reference_image_path is None and not args.dry_run:
        from agents.reference_selector import select_reference
        session_dir = args.artifact.parent.parent
        sel = select_reference(session_dir=session_dir)
        print(f"Auto-selected reference: {sel['source_filename']}  (score={sel['score']:.1f})")
        print("  Override with --reference <image_path> if this is wrong.")
        # Reference selector returns source_filename; the stripped image must exist
        # in the catalog stripped dir or the user must supply --reference explicitly.
        # We store the selection but the client will raise clearly if the file is missing.
        reference_image_path = session_dir / "stripped" / sel["source_filename"]

    run_calibration(
        target_artifact_path=args.artifact,
        run_id=args.run_id,
        batch_size=args.batch,
        seed=args.seed,
        dry_run=args.dry_run,
        prior_tuning=prior_tuning,
        generator_url=args.url,
        guidance_scale=args.guidance,
        tuner=tuner,
        reference_image_path=reference_image_path,
        ip_adapter_scale=args.ip_scale,
        model=args.model,
        resolution=args.resolution,
        generation_fn=run_generation,
    )


if __name__ == "__main__":
    main()
