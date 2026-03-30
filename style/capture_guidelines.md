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

---

## Full-Body Multi-Angle Capture Protocol

### Camera Setup & Geometry
- **Mounting & Axis:** Tripod-mounted at subject mid-thigh height (~1.0 m). Optical axis parallel to ground plane; ensure zero roll/pitch.
- **Distance & Framing:** Adjust working distance so the subject (floor contact point to crown) occupies 60–80% of the vertical frame height. Use 35–50mm (full-frame equiv.) focal length to suppress perspective distortion.
- **Required Angles:** Minimum three captures per session at consistent distance/zoom:
  - `0°` frontal
  - `45°` left-oblique
  - `45°` right-oblique
- **Metadata:** Log focal length, sensor size, and working distance in EXIF/custom tags.

### Calibration Target
- Deploy a high-contrast, known-dimension reference object (e.g., 200×200 mm matte checkerboard or ArUco marker) vertically positioned on the ground plane.
- Target must lie adjacent to the subject's lead foot, fully visible, unobstructed, and coplanar with the subject's anterior-posterior midline.

### Subject Stance
- **Posture:** Neutral anatomical stance; feet shoulder-width, weight evenly distributed, gaze forward, cervical spine aligned.
- **Arms:** Abducted ~10–15° from the torso; elbows extended, palms neutral/palms facing medially. No limb-to-body contact, clothing bunching, or floor contact with hands.

### Lighting Specifications
- **Primary Illumination:** Broad, diffused frontal light to minimize high-contrast occlusions and specular surface reflections.
- **Floor Shadow Management:** Position key/fill lights at elevated angles or use overhead diffusion to suppress hard cast shadows at the foot-floor boundary.
- **Uniformity:** Maintain fill lighting at ±30° from the camera axis to keep torso-to-extremity luminance ratio ≤ 2:1. Avoid backlighting or floor gradients that interfere with ground-plane segmentation.

### Acceptance Criteria
| Metric | Threshold | Verification |
|--------|-----------|--------------|
| Body Pose Estimation Confidence | ≥0.85 | Per-frame model output (MediaPipe PoseLandmarker) |
| Landmark Visibility Coverage | ≥0.90 (≥30/33 keypoints) | Automated keypoint mask + boundary check for head, shoulders, wrists, hips, knees, ankles |
| Height Estimable Flag | `TRUE` | Floor-plane intersection detected AND calibration target fully visible in frame |

### Pipeline Limitations
- **Height Estimation Reliability:** Metric outputs are currently unstable without an in-frame reference object of known dimensions. Sequences missing the calibration target will return `FALSE` for the `Height Estimable Flag` and are unsuitable for ground-truth labeling.
- **IOD Fallback Calibration:** Inter-Ocular Distance (IOD) scaling provides only rough proxy estimates (±5–10 cm error margin) due to lens distortion and pose variance. Do not use IOD-derived heights as primary ground truth; restrict to quality monitoring or fallback diagnostics only.
