#!/usr/bin/env python3
"""Validate artifact JSON files against a JSON Schema for CI pipelines."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

try:
    import jsonschema
except ImportError:
    print("ERROR: 'jsonschema' is required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CI script to validate artifact JSON files against a specified schema."
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("catalog"),
        help="Root directory containing session artifacts (default: catalog)"
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("anatomy/landmark_schema.json"),
        help="Path to the JSON Schema file (default: anatomy/landmark_schema.json)"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Specific session ID to validate. If omitted, validates all sessions."
    )
    return parser.parse_args()


def load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.is_file():
        print(f"ERROR: Schema file not found: {schema_path.resolve()}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Schema file contains malformed JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: Unable to read schema file: {e}", file=sys.stderr)
        sys.exit(1)


def discover_artifacts(catalog_dir: Path, session_id: Optional[str]) -> List[tuple[Path, str]]:
    artifacts = []
    session_dirs = []

    if session_id:
        target = catalog_dir / "sessions" / session_id
        if target.is_dir():
            session_dirs.append((target, session_id))
        else:
            print(f"WARNING: Session directory not found: {target}", file=sys.stderr)
    else:
        root = catalog_dir / "sessions"
        if root.is_dir():
            session_dirs = sorted(
                [(d, d.name) for d in root.iterdir() if d.is_dir()],
                key=lambda x: x[1]
            )

    for s_dir, s_name in session_dirs:
        art_dir = s_dir / "artifacts"
        if art_dir.is_dir():
            for json_file in sorted(art_dir.glob("*.json")):
                artifacts.append((json_file, s_name))
    return artifacts


def validate_artifact(file_path: Path, session_id: str, schema: Dict[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {
        "session": session_id,
        "artifact_id": file_path.stem,
        "status": "PASS",
        "error": ""
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["status"] = "FAIL"
        result["error"] = f"Malformed JSON: {e}"
        return result
    except OSError as e:
        result["status"] = "FAIL"
        result["error"] = f"IO Error: {e}"
        return result

    if isinstance(data, dict) and "artifact_id" in data:
        result["artifact_id"] = data["artifact_id"]

    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        result["status"] = "FAIL"
        path_str = ".".join(map(str, e.absolute_path)) if e.absolute_path else "root"
        result["error"] = f"Validation failed at {path_str}: {e.message}"
    except jsonschema.SchemaError as e:
        result["status"] = "FAIL"
        result["error"] = f"Invalid schema structure: {e.message}"

    return result


def main() -> int:
    args = parse_args()
    schema = load_schema(args.schema)
    artifacts = discover_artifacts(args.catalog_dir, args.session)

    if not artifacts:
        print("INFO: No artifact JSON files found to validate.")
        return 0

    results = [validate_artifact(path, session, schema) for path, session in artifacts]

    print(f"\n{'Session':<32} | {'Artifact ID':<36} | {'Status':<6} | {'Error'}")
    print("-" * 110)
    for r in results:
        err_display = r["error"] if r["error"] else "-"
        print(f"{r['session']:<32} | {r['artifact_id']:<36} | {r['status']:<6} | {err_display}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\nResults: {passed} passed, {failed} failed (Total: {len(results)})")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
