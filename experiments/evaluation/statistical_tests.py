"""Paired non-parametric comparisons across the same random-seed splits."""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_wilcoxon(
    results: pd.DataFrame,
    *,
    metric: str,
    reference_model: str = "TabPFN:v3.5",
) -> pd.DataFrame:
    """Compare each successful model with a reference using paired seed scores."""
    successful = results.loc[results["status"] == "ok"]
    if metric not in successful:
        raise ValueError(f"Metric {metric!r} is not present in the result table.")

    paired = successful.pivot_table(
        index="seed", columns="model_label", values=metric, aggfunc="mean"
    )
    if reference_model not in paired.columns:
        raise ValueError(f"Reference model {reference_model!r} is not in the results.")

    rows = []
    for model in paired.columns:
        if model == reference_model:
            continue
        scores = paired[[reference_model, model]].dropna()
        if len(scores) < 2:
            continue
        reference = scores[reference_model].to_numpy(dtype=float)
        candidate = scores[model].to_numpy(dtype=float)
        try:
            statistic, p_value = wilcoxon(reference, candidate, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        rows.append(
            {
                "reference_model": reference_model,
                "model": model,
                "metric": metric,
                "paired_seeds": len(scores),
                "median_reference_minus_model": float(np.median(reference - candidate)),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    if not rows:
        return pd.DataFrame()
    comparisons = pd.DataFrame(rows)
    order = comparisons["p_value"].sort_values().index
    count = len(order)
    adjusted = []
    running_max = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, comparisons.loc[index, "p_value"] * (count - rank))
        running_max = max(running_max, corrected)
        adjusted.append((index, running_max))
    for index, value in adjusted:
        comparisons.loc[index, "p_value_holm"] = value
    return comparisons.sort_values("p_value").reset_index(drop=True)
