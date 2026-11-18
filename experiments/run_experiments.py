"""Run a seeded comparison of TabPFN versions and baseline estimators."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from experiments.analysis.dataset_analysis import analyze_dataset
from experiments.evaluation.classification_metrics import (
    calculate_classification_metrics,
)
from experiments.evaluation.memory_usage import PeakMemoryTracker
from experiments.evaluation.model_comparison import summarize_results
from experiments.evaluation.regression_metrics import calculate_regression_metrics
from experiments.evaluation.timing import elapsed_seconds, start_timer
from experiments.models.baseline_models import build_baseline_model
from experiments.models.model_config import (
    CLASSIFICATION_BASELINES,
    REGRESSION_BASELINES,
    TABPFN_VERSIONS,
)
from experiments.models.tabpfn_models import build_tabpfn_model
from experiments.preprocessing.data_loading import load_dataset
from experiments.preprocessing.data_splitting import split_dataset


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, Any]:
    """Collect the software and accelerator context for reproducibility."""
    gpu_name = None
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except (RuntimeError, AssertionError):
            pass
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "tabpfn_version": _package_version("tabpfn"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _build_estimator(
    family: str,
    name: str,
    *,
    task: str,
    version_name: str,
    device: str,
    n_estimators: int | str,
    features: pd.DataFrame,
    seed: int,
):
    if family == "TabPFN":
        return build_tabpfn_model(
            task,
            version_name,
            device=device,
            n_estimators=n_estimators,
            seed=seed,
        )
    return build_baseline_model(task, name, features=features, seed=seed)


def _run_one_model(
    *,
    family: str,
    name: str,
    version_name: str,
    task: str,
    seed: int,
    device: str,
    n_estimators: int | str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Fit, predict, score, and report errors for one model/split pair."""
    model_label = f"{family}:{version_name}" if family == "TabPFN" else f"Baseline:{name}"
    record: dict[str, Any] = {
        "model_family": family,
        "model_name": name,
        "model_version": version_name or None,
        "model_label": model_label,
        "seed": seed,
        "device_requested": device,
        "status": "error",
        "error_type": None,
        "error_message": None,
        "fit_seconds": float("nan"),
        "predict_seconds": float("nan"),
        "effective_n_estimators": None,
    }

    tracker = PeakMemoryTracker()
    try:
        with tracker:
            fit_started = start_timer()
            estimator = _build_estimator(
                family,
                name,
                task=task,
                version_name=version_name,
                device=device,
                n_estimators=n_estimators,
                features=X_train,
                seed=seed,
            )
            estimator.fit(X_train, y_train)
            record["fit_seconds"] = elapsed_seconds(fit_started)
            record["effective_n_estimators"] = getattr(
                estimator, "n_estimators_", None
            )

            predict_started = start_timer()
            predictions = estimator.predict(X_test)
            probabilities = None
            if task == "classification":
                if hasattr(estimator, "predict_proba"):
                    probabilities = estimator.predict_proba(X_test)
            record["predict_seconds"] = elapsed_seconds(predict_started)

            metric_values: dict[str, float]
            if task == "classification":
                classes = getattr(estimator, "classes_", y_train.unique())
                metric_values = calculate_classification_metrics(
                    y_test,
                    predictions,
                    probabilities=probabilities,
                    classes=classes,
                )
            else:
                metric_values = calculate_regression_metrics(y_test, predictions)

        record.update(tracker.as_megabytes())
        record.update(metric_values)
        record["status"] = "ok"
    except Exception as error:  # Keep the full comparison moving after a model fails.
        record.update(tracker.as_megabytes())
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)[:2000]
    return record


def run_benchmark(
    config_path: str | Path,
    *,
    data_path: str | Path | None = None,
    target: str | None = None,
    task: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run all configured models and return raw and summary CSV paths."""
    config_path = _resolve_path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")

    task = task or config.get("task")
    target = target or config.get("target")
    selected_data = data_path or config.get("data_path")
    if task not in {"classification", "regression"}:
        raise ValueError("Set task to 'classification' or 'regression'.")
    if not target:
        raise ValueError("Set the dataset target column in YAML or pass --target.")
    if not selected_data:
        raise ValueError("Set data_path in YAML or pass --data.")

    data_file = _resolve_path(selected_data)
    features, labels, dataset_metadata = load_dataset(
        data_file,
        target,
        categorical_columns=config.get("categorical_columns"),
    )
    if task == "regression":
        labels = pd.to_numeric(labels, errors="raise")
    dataset_metadata["task"] = task
    dataset_metadata["profile"] = analyze_dataset(features, labels)

    test_size = float(config.get("test_size", 0.2))
    seeds = [int(seed) for seed in config.get("random_seeds", [42])]
    if not seeds:
        raise ValueError("random_seeds must contain at least one integer seed.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("random_seeds must be unique to keep paired comparisons clear.")
    versions = [str(item) for item in config.get("tabpfn_versions", [])]
    baselines = [str(item) for item in config.get("baseline_models", [])]
    if not versions and not baselines:
        raise ValueError("Configure at least one TabPFN version or baseline model.")
    unknown_versions = sorted(set(versions) - set(TABPFN_VERSIONS))
    if unknown_versions:
        raise ValueError(f"Unsupported TabPFN versions: {unknown_versions}")
    supported_baselines = (
        CLASSIFICATION_BASELINES if task == "classification" else REGRESSION_BASELINES
    )
    unknown_baselines = sorted(set(baselines) - set(supported_baselines))
    if unknown_baselines:
        raise ValueError(f"Unsupported {task} baselines: {unknown_baselines}")
    metrics = [str(item) for item in config.get("metrics", [])]
    device = str(config.get("device", "auto"))
    n_estimators = config.get("n_estimators", "auto")

    result_root = _resolve_path(output_dir or "experiments/results")
    raw_dir = result_root / "raw_results"
    processed_dir = result_root / "processed_results"
    comparison_dir = result_root / "comparison_tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = raw_dir / f"run_{run_id}.csv"
    summary_path = processed_dir / f"summary_{run_id}.csv"
    comparison_path = comparison_dir / f"summary_{run_id}.csv"
    config_snapshot = raw_dir / f"run_{run_id}_config.json"
    metadata_path = raw_dir / f"run_{run_id}_metadata.json"

    config_snapshot.write_text(
        json.dumps(
            {
                **config,
                "task": task,
                "target": target,
                "data_path": str(data_file),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    run_metadata = {
        "run_id": run_id,
        "dataset": dataset_metadata,
        "environment": environment_metadata(),
        "note": (
            "Peak RAM is sampled process resident memory; accelerator memory is not "
            "included. Delta RAM is relative to each model run's starting process RSS."
        ),
    }
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
    )

    model_specs = [("TabPFN", "TabPFN", item) for item in versions]
    model_specs.extend(("Baseline", item, "") for item in baselines)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, stratified = split_dataset(
            features,
            labels,
            task=task,
            test_size=test_size,
            seed=seed,
        )
        for family, name, version_name in model_specs:
            record = _run_one_model(
                family=family,
                name=name,
                version_name=version_name,
                task=task,
                seed=seed,
                device=device,
                n_estimators=n_estimators,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            record.update(
                {
                    "run_id": run_id,
                    "dataset_name": dataset_metadata["dataset_name"],
                    "task": task,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_features": features.shape[1],
                    "stratified_split": stratified,
                }
            )
            records.append(record)
            pd.DataFrame(records).to_csv(raw_path, index=False)
            status = record["status"]
            print(f"[{status}] seed={seed} {record['model_label']}")

    result_frame = pd.DataFrame(records)
    summary = summarize_results(result_frame, metrics)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(comparison_path, index=False)
    return raw_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/configs/classification.yaml",
        help="Path to the task YAML file (relative to the repository root by default).",
    )
    parser.add_argument("--data", help="CSV path; overrides data_path in the YAML file.")
    parser.add_argument("--target", help="Target column; overrides the YAML setting.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        help="Task type; overrides the YAML setting.",
    )
    parser.add_argument(
        "--output-dir",
        help="Results directory (defaults to experiments/results).",
    )
    args = parser.parse_args()
    raw_path, summary_path = run_benchmark(
        args.config,
        data_path=args.data,
        target=args.target,
        task=args.task,
        output_dir=args.output_dir,
    )
    print(f"Raw per-model results: {raw_path}")
    print(f"Summary table: {summary_path}")
    if not pd.read_csv(raw_path)["status"].eq("ok").any():
        raise SystemExit("Every configured model failed; see error_message in the raw results.")


if __name__ == "__main__":
    main()
