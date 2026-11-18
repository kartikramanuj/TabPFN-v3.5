"""Convenience wrapper for loading and summarizing raw benchmark CSVs."""

from pathlib import Path

import pandas as pd

from experiments.evaluation.model_comparison import summarize_results


def summarize_file(path: str | Path, metrics: list[str]) -> pd.DataFrame:
    """Load one raw result file and calculate grouped summary statistics."""
    results = pd.read_csv(path)
    return summarize_results(results, metrics)
