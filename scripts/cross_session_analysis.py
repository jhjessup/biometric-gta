"""
Cross-Session Consistency Analysis
Pools artifacts from all specified sessions and computes pairwise geometric
distance to identify subject clusters and flag outliers.

Uses a subset of geometry measurements that are:
  - scale-invariant (all normalized by IOD)
  - pose-tolerant (bilateral ratios / symmetry scores, not absolute positions)
  - present in both session 001 (legacy geometry) and session 002

Usage:
    python -m scripts.cross_session_analysis [--sessions S1 S2 ...] [--out PATH]
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
CATALOG    = REPO_ROOT / "catalog" / "sessions"

# ---------------------------------------------------------------------------
# Feature vector definition
# These 14 measurements are present in both session 001 and session 002
# artifacts and are the most discriminative / pose-stable.
# ---------------------------------------------------------------------------
FEATURE_KEYS = [
    "eye_left_aspect_ratio",    # orbital shape
    "eye_right_aspect_ratio",
    "eye_width_symmetry",       # left/right eye width ratio
    "eye_height_symmetry",
    "canthal_index",            # inner/outer canthal ratio
    "nose_width_norm",          # alar width / IOD
    "nose_length_norm",         # nasion→tip / IOD
    "nasal_index",              # width/length * 100
    "mouth_to_iod_ratio",       # mouth width / IOD
    "lip_ratio",                # upper/lower lip height ratio
    "facial_index",             # face_height/face_width * 100
    "jaw_to_cheek_ratio",       # jaw width / cheekbone width
    "facial_third_mid",         # mid-face proportion
    "facial_third_lower",       # lower-face proportion
]


def _load_artifacts(session_ids: list[str]) -> list[dict]:
    """Load all artifacts from the given sessions, annotated with session + filename."""
    records = []
    for sid in session_ids:
        session_dir = CATALOG / sid / "artifacts"
        if not session_dir.exists():
            print(f"  WARNING: session {sid} not found at {session_dir}, skipping")
            continue
        for ap in sorted(session_dir.glob("*.json")):
            a = json.loads(ap.read_text())
            if "geometry" not in a:
                continue
            m = a["geometry"].get("measurements", {})
            # Skip if too many feature keys are missing
            present = sum(1 for k in FEATURE_KEYS if m.get(k) is not None)
            if present < len(FEATURE_KEYS) * 0.7:
                continue
            records.append({
                "artifact_id":    a["artifact_id"],
                "session_id":     sid,
                "source_file":    a.get("metadata", {}).get("source_filename", "unknown"),
                "quality_flags":  a.get("metadata", {}).get("quality_flags", []),
                "confidence":     a["landmarks"].get("confidence", 0),
                "measurements":   m,
                "artifact_path":  str(ap),
                "has_body":       "body_geometry" in a,
                "body_measurements": a.get("body_geometry", {}).get("measurements", {}),
            })
    return records


def _feature_vector(m: dict) -> np.ndarray:
    """Build a feature vector, substituting 1.0 for missing bilateral ratios."""
    vec = []
    for k in FEATURE_KEYS:
        v = m.get(k)
        if v is None:
            # Sensible defaults for missing values
            v = 1.0 if "ratio" in k or "symmetry" in k or "index" in k else 0.0
        vec.append(float(v))
    return np.array(vec)


def _pairwise_distances(records: list[dict]) -> np.ndarray:
    n = len(records)
    vecs = np.stack([_feature_vector(r["measurements"]) for r in records])
    # Normalise each feature to [0,1] range across the population
    col_min = vecs.min(axis=0)
    col_max = vecs.max(axis=0)
    col_range = np.where(col_max - col_min > 0, col_max - col_min, 1.0)
    vecs_norm = (vecs - col_min) / col_range
    # Euclidean distance
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vecs_norm[i] - vecs_norm[j]))
            dist[i, j] = dist[j, i] = d
    return dist


def _find_clusters(dist: np.ndarray, records: list[dict], threshold: float) -> dict:
    """
    Simple threshold-based clustering: a record belongs to the primary cluster
    if its mean distance to all other cluster members is <= threshold.
    Seed cluster = record with lowest mean distance to all others.
    """
    n = len(records)
    mean_dists = dist.mean(axis=1)
    seed = int(np.argmin(mean_dists))

    cluster = {seed}
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in cluster:
                continue
            dists_to_cluster = [dist[i, j] for j in cluster]
            if np.mean(dists_to_cluster) <= threshold:
                cluster.add(i)
                changed = True

    outliers = [i for i in range(n) if i not in cluster]
    return {"cluster": sorted(cluster), "outliers": outliers, "seed": seed}


def _body_summary(records: list[dict], indices: list[int]) -> dict:
    """Aggregate body geometry stats for a set of artifact indices."""
    heights = []
    shoulders = []
    hips = []
    for i in indices:
        bm = records[i]["body_measurements"]
        h  = bm.get("height_cm")
        sw = bm.get("shoulder_width_cm")
        hw = bm.get("hip_width_cm")
        if h  and h  > 100:  # only plausible full-body shots
            heights.append(h)
        if sw and sw > 20:   # exclude group-shot anomalies
            shoulders.append(sw)
        if hw and hw > 10:
            hips.append(hw)

    def stats(vals):
        if not vals:
            return None
        return {
            "n":    len(vals),
            "mean": round(float(np.mean(vals)), 1),
            "std":  round(float(np.std(vals)), 1),
            "min":  round(float(np.min(vals)), 1),
            "max":  round(float(np.max(vals)), 1),
        }

    return {
        "height_cm":         stats(heights),
        "shoulder_width_cm": stats(shoulders),
        "hip_width_cm":      stats(hips),
    }


def run_analysis(session_ids: list[str], out_path: Path | None = None,
                 cluster_threshold: float | None = None) -> dict:
    print(f"\nLoading artifacts from: {session_ids}")
    records = _load_artifacts(session_ids)
    print(f"  {len(records)} artifacts with sufficient geometry data")

    if len(records) < 2:
        print("  Not enough artifacts for analysis.")
        return {}

    print("Computing pairwise distance matrix...")
    dist = _pairwise_distances(records)

    # Auto-threshold: mean of all pairwise distances
    all_dists = [dist[i, j] for i in range(len(records)) for j in range(i + 1, len(records))]
    auto_threshold = float(np.mean(all_dists))
    threshold = cluster_threshold if cluster_threshold is not None else auto_threshold
    print(f"  Distance threshold: {threshold:.4f}  (auto={auto_threshold:.4f})")

    result = _find_clusters(dist, records, threshold)
    cluster_idx   = result["cluster"]
    outlier_idx   = result["outliers"]

    cluster_dists  = [dist[i, j] for i in cluster_idx for j in cluster_idx if i < j]
    cluster_mean   = float(np.mean(cluster_dists)) if cluster_dists else 0.0
    cluster_std    = float(np.std(cluster_dists))  if cluster_dists else 0.0
    cluster_max    = float(np.max(cluster_dists))  if cluster_dists else 0.0

    print(f"\n=== Results ===")
    print(f"Primary cluster : {len(cluster_idx)} artifacts")
    print(f"  pairwise mean={cluster_mean:.4f}  std={cluster_std:.4f}  max={cluster_max:.4f}")
    print(f"Outliers        : {len(outlier_idx)} artifacts")

    def artifact_label(r):
        return f"{r['source_file']}  [{r['session_id']}]"

    print(f"\nCluster members:")
    for i in cluster_idx:
        r = records[i]
        flags = ",".join(r["quality_flags"]) or "none"
        print(f"  {artifact_label(r):55s}  conf={r['confidence']:.3f}  flags={flags}")

    if outlier_idx:
        print(f"\nOutliers:")
        for i in outlier_idx:
            r = records[i]
            min_to_cluster = min(dist[i, j] for j in cluster_idx) if cluster_idx else 0
            flags = ",".join(r["quality_flags"]) or "none"
            print(f"  {artifact_label(r):55s}  min_dist_to_cluster={min_to_cluster:.4f}  flags={flags}")

    # Body geometry summary for cluster members
    body = _body_summary(records, cluster_idx)
    print(f"\nBody geometry (primary cluster, plausible values only):")
    for k, v in body.items():
        print(f"  {k}: {v}")

    # Flag artifacts in session 002 not previously seen in session 001
    s001_files = {r["source_file"] for r in records if r["session_id"] == session_ids[0]}
    new_in_s002 = [
        records[i] for i in cluster_idx
        if records[i]["session_id"] != session_ids[0]
        and records[i]["source_file"] not in s001_files
    ]
    if len(session_ids) > 1:
        print(f"\nNew images (session 002+) confirmed in primary cluster: {len(new_in_s002)}")
        for r in new_in_s002:
            print(f"  {r['source_file']}")

    # Build report
    report = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "sessions_analyzed":     session_ids,
        "total_artifacts":       len(records),
        "feature_keys":          FEATURE_KEYS,
        "cluster_threshold":     round(threshold, 4),
        "primary_cluster": {
            "size":               len(cluster_idx),
            "pairwise_mean":      round(cluster_mean, 4),
            "pairwise_std":       round(cluster_std, 4),
            "pairwise_max":       round(cluster_max, 4),
            "members": [
                {
                    "artifact_id":   records[i]["artifact_id"],
                    "session_id":    records[i]["session_id"],
                    "source_file":   records[i]["source_file"],
                    "quality_flags": records[i]["quality_flags"],
                }
                for i in cluster_idx
            ],
        },
        "outliers": [
            {
                "artifact_id":          records[i]["artifact_id"],
                "session_id":           records[i]["session_id"],
                "source_file":          records[i]["source_file"],
                "quality_flags":        records[i]["quality_flags"],
                "min_dist_to_cluster":  round(min(dist[i, j] for j in cluster_idx), 4) if cluster_idx else None,
            }
            for i in outlier_idx
        ],
        "body_geometry_summary":  body,
        "new_images_in_cluster":  [r["source_file"] for r in new_in_s002],
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Cross-session consistency analysis")
    parser.add_argument(
        "--sessions", nargs="+",
        default=["2026-03-29_shannon_001", "2026-03-29_shannon_002"],
        help="Session IDs to include",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cluster distance threshold (default: auto)")
    args = parser.parse_args()

    default_out = REPO_ROOT / "catalog" / "cross_session_report.json"
    out_path = args.out or default_out

    run_analysis(args.sessions, out_path=out_path, cluster_threshold=args.threshold)


if __name__ == "__main__":
    main()
