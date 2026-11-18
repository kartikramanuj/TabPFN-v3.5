"""Classification metrics for hard predictions and predicted probabilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    *,
    probabilities=None,
    classes=None,
) -> dict[str, float]:
    """Return common classification metrics; undefined scores are NaN."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "roc_auc_ovr_macro": float("nan"),
        "log_loss": float("nan"),
    }
    if probabilities is None or classes is None:
        return metrics

    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    try:
        if len(classes) == 2:
            binary_truth = np.asarray(y_true) == classes[1]
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(binary_truth.astype(int), probabilities[:, 1])
            )
        else:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except (ValueError, IndexError):
        pass

    try:
        metrics["log_loss"] = float(log_loss(y_true, probabilities, labels=classes))
    except ValueError:
        pass
    return metrics
