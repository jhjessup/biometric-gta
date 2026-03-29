# Biometric GTA — Project Progress

## Status: Active — Awaiting Manual QA

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

### 2026-03-29 — Environment & All Three Tiers
- [x] Python 3.12 virtual environment created (`.venv/`)
- [x] All dependencies installed (opencv-python, mediapipe, pillow-heif, python-dotenv)

**Tier 1 — Anatomy**
- [x] `anatomy/landmark_schema.json` — JSON Schema for GTA artifacts (478-point FaceMesh, confidence, bounding box, quality flags, approval state)

**Tier 2 — Style**
- [x] `style/capture_guidelines.md` — Camera, lighting, pose, EXIF, acceptance thresholds
- [x] `style/annotation_standards.md` — Coordinate system, key landmark index table, quality flag definitions, versioning
- [x] `style/qa_checklist.md` — Pre-processing, landmark validation, artifact integrity, final approval steps

**Tier 3 — Agents**
- [x] `agents/exif_stripper.py` — Strips all EXIF via Pillow + pillow-heif (HEIC support), returns SHA-256 hash
- [x] `agents/landmark_validator.py` — Offline MediaPipe FaceLandmarker (CPU), multi-scale detection + Haar crop fallback
- [x] `agents/pipeline.py` — End-to-end orchestrator: strip → validate → save artifact

### 2026-03-29 — Catalog Infrastructure + First Session
- [x] Content-addressed catalog structure: `catalog/sessions/<session_id>/`
- [x] `scripts/ingest_session.py` — batch ingestor, writes session manifest + artifacts
- [x] `scripts/rebuild_index.py` — regenerates `catalog/index.json` + `data/catalog.db`
- [x] `scripts/setup_models.sh` — downloads MediaPipe FaceLandmarker model

**Session `2026-03-29_shannon_001` (subject: `subj_461f0955`)**
- 16 source images (HEIC + JPG), all EXIF-stripped and SHA-256 hashed
- 14 artifacts generated (confidence=1.0 on all)
  - 8 via direct MediaPipe detection
  - 6 via Haar cascade crop fallback (`face_crop_used` flag)
  - 4 of those also flagged `multiple_faces_detected` (group shots)
- 2 unrecoverable (`IMG_4664`, `IMG_5357`): high-density crowd shots, flagged `no_single_subject_detectable`

### 2026-03-29 — Consistency Analysis
- [x] Landmark geometry descriptor built (12 key points, normalized by inter-ocular distance)
- [x] Pairwise distance matrix computed across all 14 artifacts
- [x] **13-image cluster confirmed consistent** — mean pairwise dist=0.67, std=0.25, max=1.27
- [x] **`IMG_2809.JPG` flagged `subject_mismatch_suspected`** — min dist to cluster=1.74 (2.6× cluster mean); face geometry (pose_y=0.941 vs cluster range 0.48–0.74) inconsistent with primary subject
- [x] `catalog/sessions/2026-03-29_shannon_001/consistency_report.json` written

---

## In Progress

- [ ] Manual QA review — visually confirm `IMG_2809.JPG` mismatch, set `approved: true` on 13-image cluster

---

## Upcoming

- [ ] Unit tests for EXIF stripper and landmark validator
- [ ] CI: JSON schema validation on artifact output
- [ ] Second session ingest (additional subjects or captures)

---

## Notes

- All MediaPipe inference runs **locally/offline** — no network calls during validation.
- Raw biometric data lives outside the repo; paths configured via `.env`.
- `data/catalog.db` is gitignored — rebuild anytime with `python -m scripts.rebuild_index`.
- Detection pipeline is two-tier: full-image multi-scale first, Haar crop fallback second.
