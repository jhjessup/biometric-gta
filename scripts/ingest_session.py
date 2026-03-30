"""
Session Ingestor
Processes a folder of raw source images through the GTA pipeline and
writes a session manifest + artifacts to catalog/sessions/.

Usage:
    python -m scripts.ingest_session <source_dir> <session_id> [--subject-id SUBJ]
"""

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff"}
REPO_ROOT = Path(__file__).parent.parent


def anonymize_subject(name: str) -> str:
    """Derive a stable anonymized subject ID from a plain name."""
    import hashlib
    return "subj_" + hashlib.sha256(name.lower().encode()).hexdigest()[:8]


def ingest_session(source_dir: Path, session_id: str, subject_id: str) -> dict:
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from agents.exif_stripper import strip_exif_to_file, verify_no_exif
    from agents.landmark_validator import save_artifact, validate_landmarks
    from agents.geometry_analyzer import analyze_geometry
    from agents.pose_validator import validate_pose
    from agents.body_analyzer import analyze_body

    session_dir = REPO_ROOT / "catalog" / "sessions" / session_id
    artifacts_dir = session_dir / "artifacts"
    stripped_dir = session_dir / "stripped"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stripped_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in source_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not images:
        print(f"No supported images found in {source_dir}")
        sys.exit(1)

    print(f"Session: {session_id}")
    print(f"Subject: {subject_id}")
    print(f"Images:  {len(images)} found in {source_dir}\n")

    source_hashes = []
    artifact_ids = []
    failed = []

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")

        try:
            # Stage 1: Strip EXIF
            stripped_path = stripped_dir / f"{img_path.stem}_stripped.png"
            sha256 = strip_exif_to_file(img_path, stripped_path)
            assert verify_no_exif(stripped_path), "EXIF verification failed"
            source_hashes.append(f"sha256:{sha256}")
            print(f"         stripped  sha256:{sha256[:16]}...")

            # Stage 2: Validate landmarks
            artifact = validate_landmarks(stripped_path, sha256)
            artifact["metadata"]["subject_id"] = subject_id
            artifact["metadata"]["session_id"] = session_id
            artifact["metadata"]["source_filename"] = img_path.name

            flags = artifact["metadata"]["quality_flags"]
            conf = artifact["landmarks"]["confidence"]
            print(f"         confidence={conf:.3f}  flags={flags or 'none'}")

            # Stage 3: Facial geometry
            try:
                artifact["geometry"] = analyze_geometry(artifact)
                fi = artifact["geometry"]["measurements"].get("facial_index", "?")
                sym = artifact["geometry"]["measurements"].get("symmetry_index", "?")
                print(f"         geometry OK  facial_index={fi}  symmetry={sym}")
            except Exception as e:
                print(f"         geometry SKIP: {e}")

            # Stage 4: Body pose (best-effort — head/shoulder images return None)
            try:
                pose = validate_pose(stripped_path)
                if pose:
                    artifact["body_pose"] = pose
                    cov = [k for k, v in pose["coverage"].items() if v]
                    print(f"         pose OK  coverage={cov}")
                    # Stage 5: Body geometry (requires pose)
                    try:
                        artifact["body_geometry"] = analyze_body(artifact)
                        h = artifact["body_geometry"]["measurements"].get("height_cm", "?")
                        sw = artifact["body_geometry"]["measurements"].get("shoulder_width_cm", "?")
                        print(f"         body OK  height={h}cm  shoulder={sw}cm")
                    except Exception as e:
                        print(f"         body_geometry SKIP: {e}")
                else:
                    print(f"         pose: no full-body pose detected")
            except Exception as e:
                print(f"         pose SKIP: {e}")

            # Stage 6: Save artifact
            artifact_path = artifacts_dir / f"{artifact['artifact_id']}.json"
            save_artifact(artifact, artifact_path)
            artifact_ids.append(artifact["artifact_id"])

        except Exception as e:
            print(f"         FAILED: {e}")
            failed.append({"file": img_path.name, "error": str(e)})

    # Write session manifest
    manifest = {
        "session_id": session_id,
        "subject_id": subject_id,
        "capture_date": datetime.now(timezone.utc).date().isoformat(),
        "source_dir": str(source_dir),
        "source_image_count": len(images),
        "source_image_hashes": source_hashes,
        "artifacts": artifact_ids,
        "failed": failed,
        "pipeline_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": "",
    }

    manifest_path = session_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nSession complete.")
    print(f"  Artifacts: {len(artifact_ids)} succeeded, {len(failed)} failed")
    print(f"  Manifest:  {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Ingest a source image folder into the GTA catalog.")
    parser.add_argument("source_dir", type=Path, help="Folder of raw source images")
    parser.add_argument("session_id", nargs="?", help="Session ID (auto-generated if omitted)")
    parser.add_argument("--subject-id", default=None, help="Anonymized subject ID or plain name to hash")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)

    session_id = args.session_id or (
        datetime.now(timezone.utc).strftime("%Y-%m-%d") + "_" + uuid.uuid4().hex[:6]
    )

    # Derive subject_id from folder name if not provided
    raw_subject = args.subject_id or source_dir.name
    subject_id = (
        raw_subject if raw_subject.startswith("subj_")
        else anonymize_subject(raw_subject)
    )

    ingest_session(source_dir, session_id, subject_id)


if __name__ == "__main__":
    main()
