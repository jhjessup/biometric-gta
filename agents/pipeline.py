"""
GTA Pipeline — Full end-to-end processor
Chains: EXIF strip -> landmark validation -> artifact output
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from agents.exif_stripper import strip_exif_to_file, verify_no_exif
from agents.geometry_analyzer import analyze_geometry
from agents.landmark_validator import save_artifact, validate_landmarks

load_dotenv()


def run(input_image: str | Path, output_dir: str | Path) -> dict:
    """
    Run the full GTA pipeline on a single image.

    Args:
        input_image: Raw source image path
        output_dir: Directory to write stripped image + artifact JSON

    Returns:
        The generated GTA artifact dict
    """
    input_image = Path(input_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Strip EXIF
    stripped_path = output_dir / f"{input_image.stem}_stripped.png"
    print(f"[1/3] Stripping EXIF: {input_image.name}")
    sha256 = strip_exif_to_file(input_image, stripped_path)

    assert verify_no_exif(stripped_path), "EXIF strip verification failed — aborting."
    print(f"      SHA-256: {sha256}")

    # Stage 2: Validate landmarks
    print(f"[2/3] Validating landmarks (MediaPipe, offline)...")
    artifact = validate_landmarks(stripped_path, sha256)

    flags = artifact["metadata"]["quality_flags"]
    if flags:
        print(f"      Quality flags: {', '.join(flags)}")
    else:
        print(f"      No quality flags.")

    # Stage 3: Geometry analysis
    print(f"[3/4] Computing facial geometry (offline)...")
    artifact["geometry"] = analyze_geometry(artifact)
    fi = artifact["geometry"]["measurements"]["facial_index"]
    sym = artifact["geometry"]["measurements"]["symmetry_index"]
    print(f"      facial_index={fi}  symmetry={sym}")

    # Stage 4: Save artifact
    artifact_path = output_dir / f"{artifact['artifact_id']}.json"
    print(f"[4/4] Saving artifact: {artifact_path.name}")
    save_artifact(artifact, artifact_path)

    print(f"\nDone. Artifact ID: {artifact['artifact_id']}")
    return artifact


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m agents.pipeline <input_image> <output_dir>")
        sys.exit(1)

    result = run(sys.argv[1], sys.argv[2])
