"""Seeded train/test split helpers."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, bool]:
    """Split once so every model for a seed receives identical rows."""
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")

    stratify = None
    if task == "classification":
        counts = target.value_counts(dropna=False)
        test_rows = max(1, round(len(target) * test_size))
        train_rows = len(target) - test_rows
        if len(counts) > 1 and counts.min() >= 2 and min(test_rows, train_rows) >= len(counts):
            stratify = target

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, stratify is not None
