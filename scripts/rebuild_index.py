"""
Catalog Index Rebuilder
Walks catalog/sessions/ and rebuilds catalog/index.json + data/catalog.db from scratch.
The DB is always derivable from committed JSON — never the source of truth.

Usage:
    python -m scripts.rebuild_index
"""

import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CATALOG_DIR = REPO_ROOT / "catalog" / "sessions"
INDEX_PATH = REPO_ROOT / "catalog" / "index.json"
DB_PATH = REPO_ROOT / "data" / "catalog.db"


def load_all_sessions() -> list[dict]:
    sessions = []
    for manifest_path in sorted(CATALOG_DIR.glob("*/manifest.json")):
        sessions.append(json.loads(manifest_path.read_text()))
    return sessions


def load_artifact(session_id: str, artifact_id: str) -> dict | None:
    path = CATALOG_DIR / session_id / "artifacts" / f"{artifact_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def rebuild_json_index(sessions: list[dict]) -> dict:
    entries = []
    for s in sessions:
        for artifact_id in s.get("artifacts", []):
            artifact = load_artifact(s["session_id"], artifact_id)
            if not artifact:
                continue
            entries.append({
                "artifact_id": artifact_id,
                "session_id": s["session_id"],
                "subject_id": s["subject_id"],
                "capture_date": s["capture_date"],
                "image_hash": artifact["source"]["image_hash"],
                "confidence": artifact["landmarks"]["confidence"],
                "landmark_count": len(artifact["landmarks"]["face_mesh"]),
                "quality_flags": artifact["metadata"].get("quality_flags", []),
                "approved": artifact["metadata"].get("approved", False),
                "source_filename": artifact["metadata"].get("source_filename", ""),
                "created_at": artifact["metadata"]["created_at"],
            })

    index = {
        "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "session_count": len(sessions),
        "artifact_count": len(entries),
        "artifacts": entries,
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2))
    return index


def rebuild_sqlite(sessions: list[dict]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS quality_flags;
        DROP TABLE IF EXISTS artifacts;
        DROP TABLE IF EXISTS sessions;

        CREATE TABLE sessions (
            session_id       TEXT PRIMARY KEY,
            subject_id       TEXT,
            capture_date     TEXT,
            source_dir       TEXT,
            image_count      INTEGER,
            pipeline_version TEXT,
            created_at       TEXT,
            notes            TEXT
        );

        CREATE TABLE artifacts (
            artifact_id      TEXT PRIMARY KEY,
            session_id       TEXT REFERENCES sessions(session_id),
            subject_id       TEXT,
            image_hash       TEXT,
            confidence       REAL,
            landmark_count   INTEGER,
            approved         INTEGER,
            source_filename  TEXT,
            created_at       TEXT
        );

        CREATE TABLE quality_flags (
            artifact_id TEXT REFERENCES artifacts(artifact_id),
            flag        TEXT
        );

        CREATE INDEX idx_artifacts_session  ON artifacts(session_id);
        CREATE INDEX idx_artifacts_subject  ON artifacts(subject_id);
        CREATE INDEX idx_artifacts_approved ON artifacts(approved);
        CREATE INDEX idx_flags_artifact     ON quality_flags(artifact_id);
    """)

    for s in sessions:
        cur.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
            (
                s["session_id"], s["subject_id"], s["capture_date"],
                s.get("source_dir", ""), s.get("source_image_count", 0),
                s.get("pipeline_version", ""), s.get("created_at", ""),
                s.get("notes", ""),
            ),
        )
        for artifact_id in s.get("artifacts", []):
            artifact = load_artifact(s["session_id"], artifact_id)
            if not artifact:
                continue
            cur.execute(
                "INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id, s["session_id"], s["subject_id"],
                    artifact["source"]["image_hash"],
                    artifact["landmarks"]["confidence"],
                    len(artifact["landmarks"]["face_mesh"]),
                    int(artifact["metadata"].get("approved", False)),
                    artifact["metadata"].get("source_filename", ""),
                    artifact["metadata"]["created_at"],
                ),
            )
            for flag in artifact["metadata"].get("quality_flags", []):
                cur.execute("INSERT INTO quality_flags VALUES (?,?)", (artifact_id, flag))

    con.commit()
    con.close()


def main():
    print("Rebuilding catalog index...")
    sessions = load_all_sessions()
    print(f"  Sessions found: {len(sessions)}")

    index = rebuild_json_index(sessions)
    print(f"  catalog/index.json: {index['artifact_count']} artifacts")

    rebuild_sqlite(sessions)
    print(f"  data/catalog.db: rebuilt")
    print("Done.")


if __name__ == "__main__":
    main()
