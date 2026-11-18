"""CSV loading and basic validation for benchmark datasets."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataset(
    path: str | Path,
    target: str,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load a CSV, remove rows without labels, and return dataset metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    frame = pd.read_csv(path)
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset has duplicate column names: {duplicates}")
    if target not in frame.columns:
        raise ValueError(
            f"Target column {target!r} is missing. Available columns: "
            f"{frame.columns.tolist()}"
        )

    original_rows = len(frame)
    frame = frame.dropna(subset=[target]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset has no rows with a non-missing target.")

    y = frame.pop(target)
    if frame.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")

    requested_categoricals = categorical_columns or []
    missing_categoricals = sorted(set(requested_categoricals) - set(frame.columns))
    if missing_categoricals:
        raise ValueError(f"Categorical feature columns not found: {missing_categoricals}")

    # Normalize mixed object columns so sklearn encoders see a consistent type.
    object_columns = frame.select_dtypes(include=["object"]).columns
    for column in object_columns:
        frame[column] = frame[column].astype("string")
    for column in requested_categoricals:
        frame[column] = frame[column].astype("string").astype("category")

    inferred_categoricals = frame.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    metadata = {
        "dataset_name": path.stem,
        "dataset_path": str(path.resolve()),
        "n_rows": len(frame),
        "n_rows_dropped_missing_target": original_rows - len(frame),
        "n_features": frame.shape[1],
        "target": target,
        "missing_feature_fraction": float(frame.isna().to_numpy().mean()),
        "categorical_features": len(inferred_categoricals),
        "categorical_feature_names": inferred_categoricals,
        "target_unique_values": int(y.nunique(dropna=True)),
    }
    return frame, y, metadata
