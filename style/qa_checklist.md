# QA Checklist — Ground Truth Artifact Review

Complete all items before setting `metadata.approved: true` on an artifact.

## Pre-Processing
- [ ] EXIF stripped — confirmed zero EXIF fields in output image
- [ ] Image hash recorded in artifact `source.image_hash`
- [ ] Resolution meets minimum (≥ 1920×1080)

## Landmark Validation
- [ ] MediaPipe confidence ≥ 0.85
- [ ] All 478 face mesh landmarks present (or occlusion flag set)
- [ ] Bounding box within 20–70% of frame area
- [ ] No `low_confidence` or `pose_deviation` flags without manual review sign-off

## Artifact Integrity
- [ ] `artifact_id` is a valid UUID v4
- [ ] `schema_version` matches current spec (`1.0.0`)
- [ ] `created_at` timestamp is UTC
- [ ] JSON validates against `anatomy/landmark_schema.json`

## Final Approval
- [ ] Reviewed by qualified annotator
- [ ] `metadata.approved` set to `true`
- [ ] Artifact written to designated output directory
