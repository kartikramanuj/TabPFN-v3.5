"""Baseline model pipelines with training-fold-only preprocessing."""

from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in features if is_numeric_dtype(features[column].dtype)]
    categorical = [column for column in features if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _lightgbm_estimator(task: str, seed: int):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as error:
        raise ImportError(
            "The LightGBM baseline requires the 'lightgbm' package."
        ) from error

    estimator = LGBMClassifier if task == "classification" else LGBMRegressor
    return estimator(n_estimators=300, random_state=seed, n_jobs=-1, verbosity=-1)


def build_baseline_model(
    task: str,
    name: str,
    *,
    features: pd.DataFrame,
    seed: int,
):
    """Return a scikit-learn pipeline for one configured baseline."""
    if task == "classification":
        estimators = {
            "dummy": DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(
                max_iter=2000, random_state=seed
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
        }
    elif task == "regression":
        estimators = {
            "dummy": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
        }
    else:
        raise ValueError("task must be 'classification' or 'regression'.")

    if name == "lightgbm":
        estimator = _lightgbm_estimator(task, seed)
    else:
        try:
            estimator = estimators[name]
        except KeyError as error:
            raise ValueError(f"Unknown {task} baseline: {name!r}") from error

    return Pipeline(
        [
            ("preprocess", _make_preprocessor(features)),
            ("estimator", estimator),
        ]
    )
