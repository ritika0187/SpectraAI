
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_predictions(
    y_true: list,
    y_pred: list,
    y_scores: list | None = None,
) -> dict:
    """
    Evaluate binary real/fake predictions.

    Label convention:
        0 = REAL
        1 = FAKE

    y_scores contains predicted FAKE scores/probabilities.
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) == 0:
        raise ValueError("No ground-truth labels provided.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Ground-truth labels must be 0 (real) or 1 (fake).")

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("Predictions must be 0 (real) or 1 (fake).")

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            zero_division=0,
            output_dict=True,
        ),
    }

    # ROC-AUC requires continuous scores and both classes.
    if y_scores is not None:
        y_scores = np.asarray(y_scores, dtype=float)

        if len(y_scores) != len(y_true):
            raise ValueError("y_scores and y_true must have equal lengths.")

        if not np.isfinite(y_scores).all():
            raise ValueError("y_scores contains invalid values.")

        if ((y_scores < 0) | (y_scores > 1)).any():
            raise ValueError("y_scores must be between 0 and 1.")

        if len(np.unique(y_true)) == 2:
            results["roc_auc"] = float(
                roc_auc_score(y_true, y_scores)
            )
        else:
            results["roc_auc"] = None
    else:
        results["roc_auc"] = None

    return results
