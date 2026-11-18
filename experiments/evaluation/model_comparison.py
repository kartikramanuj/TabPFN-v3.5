"""Aggregate successful runs into mean and standard-deviation tables."""

from collections.abc import Iterable

import pandas as pd


def summarize_results(
    results: pd.DataFrame,
    metrics: Iterable[str],
) -> pd.DataFrame:
    """Summarize results by model across all recorded seeds."""
    successful = results.loc[results["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame()

    numeric_columns = [
        column
        for column in [
            *metrics,
            "fit_seconds",
            "predict_seconds",
            "peak_process_ram_mb",
            "peak_process_ram_delta_mb",
        ]
        if column in successful.columns
    ]
    summary = (
        successful.groupby(
            ["model_family", "model_name", "model_version", "model_label"],
            dropna=False,
        )[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary
