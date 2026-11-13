
from pathlib import Path

import cv2

from preprocessing.face.face_detector import FaceDetector
from preprocessing.face.face_cropper import FaceCropper
from preprocessing.video.frame_extractor import FrameExtractor
from models.vision.deepfake_detector import (
    score_faces,
    summarize_scores,
)

VIDEO_PATH = Path.home() / "Desktop" / "sample.mov"

extractor = FrameExtractor(sample_count=12)
detector = FaceDetector()
cropper = FaceCropper()

frames = extractor.extract_frames(str(VIDEO_PATH))

all_crops = []

for frame_data in frames:
    frame = frame_data["image"]

    faces = detector.detect(frame)
    crops = cropper.crop(frame, faces)

    all_crops.extend(crops)

print("Frames extracted:", len(frames))
print("Face crops found:", len(all_crops))

if not all_crops:
    print("No faces detected. Cannot test classifier.")
else:
    scores = score_faces(all_crops)
    summary = summarize_scores(scores)

    for index, score in enumerate(scores, start=1):
        print(
            f"Face {index}: "
            f"Prediction={score['prediction']}, "
            f"Fake score={score['fake_probability']:.4f}"
        )

    print("\nVideo-level baseline:")
    print(summary)
