"""Names of models supported by the default experiment configurations."""

TABPFN_VERSIONS = ("v2", "v2.5", "v2.6", "v3", "v3.5", "v3.5-fast")

CLASSIFICATION_BASELINES = (
    "dummy",
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "lightgbm",
)

REGRESSION_BASELINES = (
    "dummy",
    "ridge",
    "random_forest",
    "extra_trees",
    "lightgbm",
)
