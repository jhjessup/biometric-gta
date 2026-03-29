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

- [ ] Set up Python 3.12 virtual environment (`.venv/`)
- [ ] Install dependencies from `scripts/requirements.txt`

---

## Upcoming

### Tier 1 — Anatomy (JSON Schemas)
- [ ] Define landmark schema for facial biometrics
- [ ] Define ground truth artifact format spec
- [ ] Create sample/test fixture in `/anatomy/`

### Tier 2 — Style (Standards)
- [ ] Write capture guidelines in `/style/`
- [ ] Write annotation standards doc
- [ ] Write QA checklist

### Tier 3 — Agents (Pipeline)
- [ ] EXIF stripping pre-processor (Pillow)
- [ ] MediaPipe landmark validation agent
- [ ] Ground truth artifact generator template

---

## Notes

- All MediaPipe inference must run **locally/offline** — no network calls during validation.
- Raw biometric data lives outside the repo; paths configured via `.env`.
