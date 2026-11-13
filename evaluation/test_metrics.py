
from evaluation.metrics import evaluate_predictions


def main():
    # 0 = REAL, 1 = FAKE
    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]
    y_scores = [0.10, 0.70, 0.90, 0.80, 0.20, 0.40]

    results = evaluate_predictions(y_true, y_pred, y_scores)

    print("\nSPECTRAAI — METRICS TEST")
    print("-" * 35)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]:
        print(f"{metric}: {results[metric]}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    for label, values in results["classification_report"].items():
        if isinstance(values, dict):
            print(
                f"{label}: "
                f"precision={values['precision']:.3f}, "
                f"recall={values['recall']:.3f}, "
                f"f1={values['f1-score']:.3f}"
            )


if __name__ == "__main__":
    main()
