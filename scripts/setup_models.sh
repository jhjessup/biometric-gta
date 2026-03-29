#!/usr/bin/env bash
# Download required MediaPipe model files into models/
set -e
mkdir -p "$(dirname "$0")/../models"
cd "$(dirname "$0")/../models"

echo "Downloading FaceLandmarker model..."
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "Done: $(du -sh face_landmarker.task)"
