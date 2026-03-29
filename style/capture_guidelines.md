# Image Capture Guidelines

## Camera & Environment

- **Resolution**: Minimum 1920×1080. 4K preferred for landmark precision.
- **Lighting**: Diffuse, even frontal lighting. Avoid hard shadows across the face. No backlight.
- **Background**: Neutral, non-reflective. Avoid patterns that confuse edge detection.
- **Distance**: Subject face occupies 40–70% of frame height.

## Subject Pose

- **Neutral pose**: Head level, gaze forward, mouth relaxed (closed or slightly open).
- **Pitch/yaw/roll**: Keep within ±10° of center for primary captures.
- **Occlusion**: No sunglasses, heavy makeup, or obstructions over landmark regions.

## EXIF & Privacy

- Strip ALL EXIF metadata before ingest using the EXIF stripping script.
- Never store GPS, device ID, or timestamp in the image file.
- Images are referenced only by their SHA-256 hash in GTA artifacts.

## Acceptance Criteria

| Check | Threshold |
|-------|-----------|
| MediaPipe detection confidence | ≥ 0.85 |
| Face bounding box area (% of frame) | 20–70% |
| Landmark visibility (avg) | ≥ 0.90 |
| EXIF fields present after strip | 0 |
