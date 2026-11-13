"""Pretrained vision baseline for face-crop deepfake classification."""

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "hamzenium/ViT-Deepfake-Classifier"


@lru_cache(maxsize=1)
def load_detector() -> tuple[Any, Any, torch.device]:
    """Load the image processor and model once per running application."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model, device


def score_faces(face_crops: list[np.ndarray]) -> list[dict[str, float | str]]:
    """Return real/fake predictions for each RGB face crop."""
    if not face_crops:
        return []

    processor, model, device = load_detector()
    inputs = processor(images=face_crops, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()

    labels = {
        int(index): str(label).upper()
        for index, label in model.config.id2label.items()
    }
    fake_index = next(
        (index for index, label in labels.items() if "FAKE" in label),
        None,
    )

    if fake_index is None:
        raise RuntimeError(f"The model has no FAKE label: {labels}")

    results = []
    for probability_row in probabilities:
        predicted_index = int(probability_row.argmax())
        results.append(
            {
                "prediction": labels[predicted_index],
                "fake_probability": float(probability_row[fake_index]),
            }
        )

    return results


def summarize_scores(scores: list[dict[str, float | str]]) -> dict[str, float | str]:
    """Aggregate face-level predictions into a vision-only video result."""
    if not scores:
        return {
            "prediction": "No faces",
            "fake_probability": 0.0,
        }

    fake_probability = float(
        np.mean([float(score["fake_probability"]) for score in scores])
    )
    prediction = "Likely fake" if fake_probability >= 0.5 else "Likely real"

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
    }
