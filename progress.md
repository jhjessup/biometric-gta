# Biometric GTA — Project Progress

## Status: In Setup

---

## Completed

### 2026-03-29 — Initial Project Scaffold
- [x] Initialized git repo at `/root/biometric-gta/`
- [x] Published to GitHub: https://github.com/jhjessup/biometric-gta
- [x] Created `CLAUDE.md` with 3-tier architecture and project rules
- [x] Established directory structure: `/anatomy/`, `/style/`, `/agents/`
- [x] Created `scripts/requirements.txt` (opencv-python, mediapipe, Pillow, python-dotenv)
- [x] Configured `.claude/settings.local.json` (auto-mode ON, notify on `rm`/network commands)
- [x] Added `.gitignore` (excludes `.venv/`, `.env`, raw biometric data)
- [x] Added `.env.example` template

---

## In Progress

- [ ] End-to-end pipeline test with a real image
- [ ] Unit tests for EXIF stripper and landmark validator

---

## Completed (continued)

### 2026-03-29 — Environment & All Three Tiers

- [x] Python 3.12 virtual environment created (`.venv/`)
- [x] All dependencies installed (opencv-python, mediapipe, Pillow, python-dotenv)

**Tier 1 — Anatomy**
- [x] `anatomy/landmark_schema.json` — JSON Schema for GTA artifacts (478-point FaceMesh, confidence, bounding box, quality flags, approval state)

**Tier 2 — Style**
- [x] `style/capture_guidelines.md` — Camera, lighting, pose, EXIF, acceptance thresholds
- [x] `style/annotation_standards.md` — Coordinate system, key landmark index table, quality flag definitions, versioning
- [x] `style/qa_checklist.md` — Pre-processing, landmark validation, artifact integrity, final approval steps

**Tier 3 — Agents**
- [x] `agents/exif_stripper.py` — Strips all EXIF via Pillow, returns SHA-256 hash, verifies clean output
- [x] `agents/landmark_validator.py` — Offline MediaPipe FaceMesh inference, produces GTA artifact JSON
- [x] `agents/pipeline.py` — End-to-end orchestrator: strip → validate → save artifact

---

## Upcoming

- [ ] `scripts/test_pipeline.py` — integration test with fixture image
- [ ] `anatomy/` — sample fixture artifact for schema validation testing
- [ ] CI: JSON schema validation on artifact output

---

## Notes

- All MediaPipe inference must run **locally/offline** — no network calls during validation.
- Raw biometric data lives outside the repo; paths configured via `.env`.
