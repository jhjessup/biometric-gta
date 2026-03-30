"""
Reference Selector
Scores GTA session artifacts and returns the best candidate for use as
the InstantID face reference image.

Scoring is entirely derived from existing artifact data — no re-processing.

Criteria (higher = better reference):
  + landmarks.confidence    — stronger detection = cleaner embedding
  + symmetry_index (low)    — more symmetric face = more stable identity embedding
  + enrichment present      — fully processed artifact preferred
  - face_crop_used          — Haar-crop fallback = less reliable alignment
  - multiple_faces_detected — group shot, wrong embedding risk
  - pose_deviation          — off-axis face degrades identity encoding
  - low_confidence          — weak detection
  - subject_mismatch_suspected — wrong subject entirely
"""

import json
from pathlib import Path


_PENALTIES: dict[str, float] = {
    "face_crop_used":              5.0,
    "multiple_faces_detected":    15.0,
    "pose_deviation":             10.0,
    "low_confidence":             10.0,
    "partial_occlusion":           8.0,
    "subject_mismatch_suspected": 50.0,
}


def score_artifact(artifact: dict) -> float:
    score = 0.0

    # Landmark detection confidence (0–1 → 0–30 pts)
    confidence = artifact.get("landmarks", {}).get("confidence", 0.0)
    score += confidence * 30.0

    # Facial symmetry: lower index = better; contributes up to 20 pts
    sym = artifact.get("geometry", {}).get("measurements", {}).get("symmetry_index")
    if sym is not None:
        score += max(0.0, (0.5 - sym)) * 40.0

    # Quality flag penalties
    flags = set(artifact.get("metadata", {}).get("quality_flags", []))
    for flag in flags:
        score -= _PENALTIES.get(flag, 2.0)

    # Bonus: fully enriched artifact (Gemini + hair analysis)
    enrichment = artifact.get("enrichment", {})
    if enrichment.get("forensic"):
        score += 5.0
    if enrichment.get("hair_analysis"):
        score += 3.0

    return score


def select_reference(
    session_dir: Path | None = None,
    artifact_paths: list[Path] | None = None,
) -> dict:
    """
    Select the best face reference artifact.

    Args:
        session_dir:    Session directory containing an artifacts/ subdirectory.
        artifact_paths: Explicit list of artifact JSON paths (alternative).

    Returns:
        {
          "artifact":        full artifact dict,
          "artifact_path":   Path to artifact JSON,
          "source_filename": original source filename,
          "score":           float quality score,
          "ranked":          full ranked list of dicts,
        }
    """
    if artifact_paths is None:
        if session_dir is None:
            raise ValueError("Provide either session_dir or artifact_paths")
        artifact_paths = sorted((session_dir / "artifacts").glob("*.json"))

    if not artifact_paths:
        raise ValueError("No artifact paths found")

    scored: list[tuple[float, str, Path, dict]] = []
    for path in artifact_paths:
        try:
            artifact = json.loads(Path(path).read_text())
        except Exception:
            continue
        s = score_artifact(artifact)
        source_file = artifact.get("metadata", {}).get("source_filename", path.stem)
        scored.append((s, source_file, Path(path), artifact))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_source, best_path, best_artifact = scored[0]

    return {
        "artifact":        best_artifact,
        "artifact_path":   best_path,
        "source_filename": best_source,
        "score":           best_score,
        "ranked": [
            {
                "score":           round(s, 2),
                "source_filename": sf,
                "artifact_path":   str(p),
            }
            for s, sf, p, _ in scored
        ],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.reference_selector <session_dir>")
        print("       python -m agents.reference_selector <artifact1.json> [artifact2.json ...]")
        sys.exit(1)

    first = Path(sys.argv[1])
    if first.is_dir():
        result = select_reference(session_dir=first)
    else:
        result = select_reference(artifact_paths=[Path(p) for p in sys.argv[1:]])

    print(f"Best reference: {result['source_filename']}  (score={result['score']:.1f})")
    print(f"Artifact:       {result['artifact_path']}")
    print(f"\nTop 5:")
    for r in result["ranked"][:5]:
        print(f"  {r['score']:6.1f}  {r['source_filename']}")
