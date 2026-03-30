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

### 2026-03-29 — Gemini Enrichment

**Gemini enrichment pipeline**
- [x] `agents/gemini_enricher.py` — Gemini 2.5 Flash vision enricher; resizes to 320px, structured JSON output
- [x] `anatomy/landmark_schema.json` — Added `enrichment` block (`forensic` + `sartorial` subsections) and new quality flags
- [x] `scripts/requirements.txt` — Added `google-generativeai>=0.8.0`
- [x] `.env.example` — Added `GEMINI_API_KEY=` template entry
- [x] All 13 cluster artifacts enriched (IMG_2809 mismatch excluded)
  - Consistent forensic profile: oval face, fair-medium skin, brown/dark_brown hair, smiling expression
  - Sartorial: varied (dresses, tops in multiple colors)
  - Free-tier rate limit (5 req/min) handled with 13s delays between calls

### 2026-03-29 — Consistency Analysis
- [x] Landmark geometry descriptor built (12 key points, normalized by inter-ocular distance)
- [x] Pairwise distance matrix computed across all 14 artifacts
- [x] **13-image cluster confirmed consistent** — mean pairwise dist=0.67, std=0.25, max=1.27
- [x] **`IMG_2809.JPG` flagged `subject_mismatch_suspected`** — min dist to cluster=1.74 (2.6× cluster mean); face geometry (pose_y=0.941 vs cluster range 0.48–0.74) inconsistent with primary subject
- [x] `catalog/sessions/2026-03-29_shannon_001/consistency_report.json` written

---

### 2026-03-29 — Full Body Biometric Pipeline + Session 002

**New agents**
- [x] `agents/pose_validator.py` — MediaPipe PoseLandmarker (33-point 3D), multi-scale detection, coverage flags
- [x] `agents/body_analyzer.py` — Anthropometric measurements from pose landmarks: shoulder/hip/waist/chest width, torso + arm + leg lengths, height estimate, body ratios, posture indicators; cm calibration via IOD population mean (63mm)
- [x] `scripts/setup_models.sh` — Added `pose_landmarker.task` download
- [x] `scripts/ingest_session.py` — Added stages 3–5: facial geometry, pose validation, body geometry
- [x] `models/pose_landmarker.task` — Downloaded (9.4 MB)

**Session `2026-03-29_shannon_002` (18 new images + 16 reprocessed)**
- 34 source images total (all files in source_images_shannon)
- 31 artifacts generated, 3 failed (IMG_2281, IMG_4664, IMG_5357 — crowd shots, no isolatable subject)
- Body pose detected in all 31; coverage breakdown:
  - 10 with full legs/feet visible (height estimable)
  - 3 with lower body partial
  - 18 head/shoulder only
- Height estimates range 32–165 cm; only `IMG_3929.HEIC` (165.5 cm) plausible for single-subject full-body
- Shoulder widths 21–52 cm for clean single-subject shots; group/anomalous shots excluded
- `lp_image.HEIC` flagged anomalous (facial_index=291, shoulder=152 cm) — likely non-standard orientation
- 10 Gemini enrichments completed; 21 pending (free-tier daily cap of 20 req/day hit)

---

### 2026-03-29 — Cross-Session Consistency Analysis

- [x] `scripts/cross_session_analysis.py` — pools all sessions, 14-feature geometry descriptor, pairwise distance matrix, auto-threshold clustering, body geometry summary
- [x] `catalog/cross_session_report.json` — analysis across 45 artifacts (sessions 001 + 002)

**Findings:**
- **38/45 artifacts** form a consistent primary cluster (pairwise mean=0.661, max=1.331)
- **12 new images** from session 002 confirmed in cluster
- **`IMG_2809.JPG` re-evaluated** — face geometry is identical across both sessions (delta <0.02); session 001 mismatch flag was based on pose position, not face geometry. Recommend visual QA to resolve.
- **True outliers (7):**
  - `lp_image.HEIC` (dist=2.26) — non-frontal/profile image, invalid biometric artifact
  - `lp_image(2).HEIC` (dist=1.29) — non-frontal, same
  - `IMG_4866.HEIC` (dist=0.93) — possibly different subject (divergent nasal + canthal index)
  - `IMG_4720.HEIC` (dist=0.59, both sessions) — borderline; consistent geometry, likely population-shift effect
  - `IMG_7584.JPEG`, `IMG_7757.JPEG` — borderline outliers for further review
- **Body geometry (primary cluster):** shoulder width mean=43.9cm ±7.4; height estimable from 2 full-body shots (126–165 cm)

---

## In Progress

- [ ] Gemini enrichment — 21 artifacts in session 002 pending (daily quota reset required)
- [ ] Manual QA review — resolve `IMG_2809.JPG` mismatch flag; confirm `lp_image` files as non-frontal; set `approved: true` on primary cluster
- [ ] Re-strip all 16 images with ICC-profile-removing `exif_stripper.py` and update artifact `source.image_hash` values

---

### 2026-03-29 — Synthesizer Calibration Layer

- [x] `agents/prompt_builder.py` — translates GTA artifact → weighted SD prompt pair; 7 descriptor lookup tables mapped from geometry measurements; sartorial block from enrichment; optics constants
- [x] `scripts/perchance_driver.py` — Playwright automation wrapper for perchance.org; fills prompt/negative/style/seed/batch fields; downloads generated images; `--dump-selectors` debug mode
- [x] `scripts/calibration_loop.py` — full calibration loop: build prompt → generate → measure synthetic → delta vs. GT → tuning suggestions; `--dry-run` and `--tune` flags for iteration
- [x] Fixed `canthal_tilt_right_deg` bug in `geometry_analyzer.py` (was returning ~180° due to mirrored x-direction); recomputed all 45 artifacts
- [x] `scripts/requirements.txt` — added `playwright>=1.40.0`
- Calibration target: `IMG_5140.HEIC` (session 001) — highest symmetry (0.043), enriched, no quality flags
- Dry-run verified: prompt generates correctly from artifact

**To run first live calibration:**
```bash
pip install playwright && playwright install chromium
python -m scripts.calibration_loop catalog/sessions/2026-03-29_shannon_001/artifacts/539144fb-7a3e-4392-b810-6de869005408.json --batch 3
```

---

## In Progress

- [ ] Gemini enrichment — 21 artifacts in session 002 pending (daily quota reset required)
- [ ] Manual QA review — resolve `IMG_2809.JPG` mismatch flag; confirm `lp_image` files as non-frontal; set `approved: true` on primary cluster
- [ ] Re-strip all 16 images with ICC-profile-removing `exif_stripper.py` and update artifact `source.image_hash` values

## Upcoming

- [ ] Live calibration run — install playwright, execute first generation, iterate on delta
- [ ] Update `style/capture_guidelines.md` with multi-angle full-body capture protocol
- [ ] Unit tests for EXIF stripper and landmark validator
- [ ] CI: JSON schema validation on artifact output
- [ ] Height calibration improvement — multi-angle or known-reference approach

---

## Notes

- All MediaPipe inference runs **locally/offline** — no network calls during validation.
- Raw biometric data lives outside the repo; paths configured via `.env`.
- `data/catalog.db` is gitignored — rebuild anytime with `python -m scripts.rebuild_index`.
- Detection pipeline is two-tier: full-image multi-scale first, Haar crop fallback second.
