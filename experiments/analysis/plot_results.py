"""Plot model metric means and standard deviations from a raw run CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(
    results_path: str | Path,
    metric: str,
    output_path: str | Path,
    *,
    higher_is_better: bool = True,
) -> Path:
    """Save a bar chart for one metric across successful models and seeds."""
    results = pd.read_csv(results_path)
    if metric not in results.columns:
        raise ValueError(f"Metric {metric!r} is missing from {results_path}.")
    successful = results.loc[results["status"] == "ok", ["model_label", metric]].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful.dropna(subset=[metric])
    if successful.empty:
        raise ValueError(f"No successful rows contain a value for {metric!r}.")

    summary = successful.groupby("model_label")[metric].agg(["mean", "std"])
    summary = summary.sort_values("mean", ascending=not higher_is_better)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(max(8, len(summary) * 1.1), 5))
    axis.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
    )
    axis.set_title(f"{metric} by model")
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
