# Biometric Ground Truth Artifact (GTA) — Claude Project Rules

## Architecture: 3-Tier Structure

```
/anatomy/    — Ground truth data definitions (JSON format)
/style/      — Visual and formatting standards (Markdown)
/agents/     — Processing pipeline templates (Templates)
```

### Tier Details

| Tier | Path | Format | Purpose |
|------|------|--------|---------|
| Anatomy | `/anatomy/` | JSON | Landmark schemas, biometric data specs, ground truth definitions |
| Style | `/style/` | Markdown | Capture guidelines, annotation standards, QA checklists |
| Agents | `/agents/` | Templates | Pipeline configs, processing workflows, validation scripts |

---

## Tech Stack

- **Python**: 3.12+ required. All scripts must target `>=3.12`.
- **Virtual env**: Always use `.venv/` in the project root. Never install packages globally.
  - Activate: `source .venv/bin/activate`
  - Create: `python3.12 -m venv .venv`
- **Image pre-processing**: Use `Pillow` for all local image operations including EXIF stripping.
- **Landmark validation**: Use `MediaPipe` for local landmark detection and validation.
- **Dependencies**: Managed via `scripts/requirements.txt`.

---

## Workflow: Plan-then-Execute (REQUIRED)

All non-trivial work must follow this sequence:

1. **Plan** — Write out the approach, files to change, and expected outcomes before touching code.
2. **Confirm** — Surface the plan to the user; wait for approval on destructive or ambiguous steps.
3. **Execute** — Implement exactly what was planned. No scope creep.
4. **Verify** — Run relevant tests or validation scripts before marking complete.

> Do not skip to Execute. Do not modify files outside the stated plan scope.

---

## Security & Privacy Rules

- Strip ALL EXIF metadata from images before any processing or storage. Use `Pillow` (`piexif` or `Image.info` clearing).
- Never log or store biometric raw data in plaintext outside of designated `/anatomy/` JSON schemas.
- No network calls during landmark validation — all MediaPipe inference must be local/offline.
- Secrets go in `.env` (never committed). Load with `python-dotenv`.

---

## Permissions (enforced in `.claude/settings.local.json`)

- `rm` commands: always prompt before execution.
- Network commands: always prompt before execution.
- Auto-mode is enabled for read/write/edit operations within the project.
