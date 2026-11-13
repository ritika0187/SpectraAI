"""Face detection backed by OpenCV's YuNet neural network."""

from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_detection"
    / "face_detection_yunet_2023mar.onnx"
)


class FaceDetector:
    """Detect the most prominent face in an RGB video frame."""

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        nms_threshold: float = 0.3,
    ) -> None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {MODEL_PATH}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (320, 320),
            confidence_threshold,
            nms_threshold,
            5_000,
        )

    def detect(
        self,
        image,
    ) -> list[tuple[int, int, int, int]]:
        """
        Detect the highest-confidence primary face.

        Accepts either:
        - NumPy RGB image
        - PIL Image

        Returns:
            List containing one (x, y, width, height) tuple.
        """

        # -------------------------------------------------
        # Convert PIL Image -> NumPy array if necessary
        # -------------------------------------------------

        if not isinstance(image, np.ndarray):

            try:
                image = np.asarray(image)

            except Exception:
                return []

        # -------------------------------------------------
        # Validate image
        # -------------------------------------------------

        if image is None:
            return []

        if image.ndim < 2:
            return []

        if image.shape[0] == 0 or image.shape[1] == 0:
            return []

        # -------------------------------------------------
        # Make sure image is uint8
        # -------------------------------------------------

        if image.dtype != np.uint8:

            image = np.clip(
                image,
                0,
                255
            ).astype(np.uint8)

        # -------------------------------------------------
        # Get dimensions
        # -------------------------------------------------

        height, width = image.shape[:2]

        # -------------------------------------------------
        # Set YuNet input size
        # -------------------------------------------------

        self.detector.setInputSize(
            (width, height)
        )

        # -------------------------------------------------
        # RGB -> BGR
        # -------------------------------------------------

        if image.ndim == 2:

            bgr_image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        elif image.shape[2] == 4:

            bgr_image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )

        else:

            bgr_image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        # -------------------------------------------------
        # Detect
        # -------------------------------------------------

        _, detections = self.detector.detect(
            bgr_image
        )

        if detections is None:
            return []

        candidates = []

        for detection in detections:

            x, y, box_width, box_height = (
                detection[:4]
            )

            confidence = float(
                detection[-1]
            )

            area = float(
                box_width * box_height
            )

            score = (
                area * confidence
            )

            candidates.append(
                (
                    score,
                    x,
                    y,
                    box_width,
                    box_height,
                )
            )

        if not candidates:
            return []

        (
            _,
            x,
            y,
            box_width,
            box_height,
        ) = max(
            candidates,
            key=lambda item: item[0]
        )

        # -------------------------------------------------
        # Clamp coordinates
        # -------------------------------------------------

        left = max(
            0,
            int(x)
        )

        top = max(
            0,
            int(y)
        )

        right = min(
            width,
            int(x + box_width)
        )

        bottom = min(
            height,
            int(y + box_height)
        )

        if right <= left or bottom <= top:
            return []

        return [
            (
                left,
                top,
                right - left,
                bottom - top,
            )
        ]

    def draw(
        self,
        image,
    ) -> tuple[
        np.ndarray,
        list[tuple[int, int, int, int]],
    ]:
        """
        Draw the selected face box onto an RGB image.
        """

        # Convert PIL -> NumPy if needed

        if not isinstance(image, np.ndarray):

            image = np.asarray(image)

        output = image.copy()

        faces = self.detect(
            image
        )

        for x, y, width, height in faces:

            cv2.rectangle(
                output,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                (0, 255, 0),
                2,
            )

        return output, faces