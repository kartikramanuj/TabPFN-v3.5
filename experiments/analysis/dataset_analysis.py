"""Dataset profiling helpers."""

from typing import Any

import pandas as pd


def analyze_dataset(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Return a compact profile without changing the input data."""
    categorical = features.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    return {
        "n_rows": int(len(features)),
        "n_features": int(features.shape[1]),
        "n_numeric_features": int(features.shape[1] - len(categorical)),
        "n_categorical_features": int(len(categorical)),
        "categorical_feature_names": categorical,
        "missing_feature_fraction": float(features.isna().to_numpy().mean()),
        "target_unique_values": int(target.nunique(dropna=True)),
        "target_missing_values": int(target.isna().sum()),
    }
