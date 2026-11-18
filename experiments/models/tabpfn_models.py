"""Factories for the TabPFN versions implemented in the upstream package."""

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion


def build_tabpfn_model(
    task: str,
    version: str,
    *,
    device: str = "auto",
    n_estimators: int | str = "auto",
    seed: int = 0,
):
    """Build a classifier or regressor for an explicit checkpoint version."""
    try:
        model_version = ModelVersion(version)
    except ValueError as error:
        supported = ", ".join(item.value for item in ModelVersion)
        raise ValueError(
            f"Unsupported TabPFN version {version!r}; choose from {supported}."
        ) from error

    if task == "classification":
        estimator = TabPFNClassifier
    elif task == "regression":
        estimator = TabPFNRegressor
    else:
        raise ValueError("task must be 'classification' or 'regression'.")

    return estimator.create_default_for_version(
        model_version,
        device=device,
        n_estimators=n_estimators,
        random_state=seed,
        show_progress_bar=False,
    )
