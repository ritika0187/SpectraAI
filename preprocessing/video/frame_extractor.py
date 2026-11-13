
import cv2
import numpy as np


class FrameExtractor:

    def __init__(self, sample_count=12):
        self.sample_count = sample_count

    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception("Cannot open video.")

        # Automatically apply video rotation metadata.
        orientation = 0

        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            orientation = int(
                cap.get(cv2.CAP_PROP_ORIENTATION_META)
            )

        auto_rotation = False

        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            auto_rotation = cap.set(
                cv2.CAP_PROP_ORIENTATION_AUTO,
                1
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            raise Exception("Video contains no readable frames.")

        indices = np.linspace(
            0,
            total_frames - 1,
            min(self.sample_count, total_frames),
            dtype=int
        )

        frames = []

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

            success, frame = cap.read()

            if not success:
                continue

            # Manual fallback if automatic rotation is unavailable.
            if not auto_rotation:
                if orientation == 90:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_CLOCKWISE
                    )
                elif orientation == 180:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_180
                    )
                elif orientation == 270:
                    frame = cv2.rotate(
                        frame,
                        cv2.ROTATE_90_COUNTERCLOCKWISE
                    )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append({
                "index": int(idx),
                "image": frame
            })

        cap.release()

        return frames
