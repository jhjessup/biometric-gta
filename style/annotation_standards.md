# Annotation Standards

## Coordinate System

All landmark coordinates use **normalized image space**:
- `x`, `y` in range `[0.0, 1.0]` (origin = top-left)
- `z` is depth relative to face center (negative = closer to camera)

## Landmark Index Reference (MediaPipe FaceMesh)

Key regions for GTA purposes:

| Region | Landmark Indices |
|--------|-----------------|
| Left eye | 33, 7, 163, 144, 145, 153, 154, 155, 133 |
| Right eye | 362, 382, 381, 380, 374, 373, 390, 249, 263 |
| Nose tip | 1 |
| Nose bridge | 6 |
| Left mouth corner | 61 |
| Right mouth corner | 291 |
| Chin | 152 |
| Left ear | 234 |
| Right ear | 454 |

## Quality Flag Definitions

| Flag | Trigger Condition |
|------|------------------|
| `low_confidence` | MediaPipe confidence < 0.85 |
| `partial_occlusion` | Visibility < 0.7 on ≥ 10 landmarks in a single region |
| `lighting_variance` | Pixel intensity std dev > 80 in face bounding box |
| `pose_deviation` | Estimated head pose > ±15° on any axis |
| `exif_present_stripped` | EXIF data was found and stripped (logged for audit) |

## Versioning

Each schema change increments `schema_version` per semver:
- **Patch**: Clarification, no structural change
- **Minor**: New optional fields added
- **Major**: Required fields added/removed, coordinate system change
