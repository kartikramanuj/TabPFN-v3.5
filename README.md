# TabPFN Research Benchmark

This repository combines the official TabPFN implementation with a reproducible
framework for evaluating its model versions against traditional machine learning
baselines on a tabular dataset. The upstream implementation, examples, documentation,
and tests are preserved in this repository. The experiment code is isolated in
[`experiments/`](experiments/).

## What is included

- TabPFN checkpoints: `v2`, `v2.5`, `v2.6`, `v3`, `v3.5`, and `v3.5-fast`.
- Classification and regression benchmark configurations.
- Matching seeded train/test splits across every model in a run.
- Baselines built from scikit-learn and LightGBM.
- Per-model metric, runtime, memory, and error records, plus a summary table.
- Raw datasets and generated result files are ignored by Git by default.

The experiment runner uses the version factory in the included TabPFN package. It
records a failure for a model that cannot run (for example, because its checkpoint
limit or access requirements are not met) and continues with the remaining models.

## Install

Create and activate a virtual environment, then install this checkout with the
experiment dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments]"
```

TabPFN downloads model weights on first use. Follow the upstream [model access and
license instructions](docs/TABPFN-UPSTREAM-README.md#installation--setup) before
starting a benchmark. The Python source is Apache 2.0 licensed; model weights have
separate terms. Check those terms for the versions and use case in your study.

## Add a dataset

Put a CSV file in `experiments/datasets/raw/` and identify its target column. You can
list numeric-coded categorical features in `categorical_columns` in the YAML config.
The runner drops only rows with a missing target. It leaves feature values intact for
TabPFN and fits imputation and categorical encoding inside each baseline pipeline
using that split's training data.

For classification:

```bash
python -m experiments.run_experiments `
  --config experiments/configs/classification.yaml `
  --data experiments/datasets/raw/your_dataset.csv `
  --target target_column
```

For regression:

```bash
python -m experiments.run_experiments `
  --config experiments/configs/regression.yaml `
  --data experiments/datasets/raw/your_dataset.csv `
  --target target_column
```

The command-line syntax above uses PowerShell line continuation. Run the command on
one line in other shells, or use `\` for line continuation in Bash.

## Configure the comparison

Edit the relevant YAML file to set the test fraction, random seeds, TabPFN versions,
baseline names, device, and ensemble size. The default comparison includes every
version exposed by the included `ModelVersion` enum and these baselines:

- Classification: dummy prior, logistic regression, random forest, extra trees,
  LightGBM.
- Regression: dummy mean, ridge, random forest, extra trees, LightGBM.

Every model uses the same train/test split for each seed. The baseline transforms
are fitted only on training rows. TabPFN receives the original feature columns and
uses its own preprocessing. Default estimator settings are used for each checkpoint;
the selected checkpoint, configured seed, and measured runtimes are recorded.

## Results

Each run writes a raw CSV and its JSON configuration snapshot under
`experiments/results/raw_results/`, and an aggregated CSV under
`experiments/results/processed_results/` and `experiments/results/comparison_tables/`.
The raw file includes successful and
failed model rows. It records classification metrics (accuracy, balanced accuracy,
macro F1, ROC AUC where defined, and log loss) or regression metrics (RMSE, MAE, and
R-squared), fit and prediction time, peak process RAM, and change from starting RAM.
A first-use checkpoint
download can affect fit time. The run metadata records the software and device context.

The Python helpers in `experiments/evaluation/` can summarize runs and perform paired
Wilcoxon comparisons across the recorded seeds. See
[`experiments/README.md`](experiments/README.md) for module details and the data
handling assumptions.

## Source and attribution

The TabPFN package code, license, notices, changelog, examples, and tests are retained
from the supplied `TabPFN-main` reference checkout (the reference `pyproject.toml`
reports package version `10.2.0`). The original upstream README is available at
[`docs/TABPFN-UPSTREAM-README.md`](docs/TABPFN-UPSTREAM-README.md).
