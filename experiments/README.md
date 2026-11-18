# Experiment framework

The framework compares explicit model checkpoints from the included TabPFN source
with classical baselines. Keep one installed TabPFN code version for a run and select
each checkpoint through `ModelVersion`; this keeps the Python preprocessing and API
fixed while the model version changes.

## Data handling

- Input is a local CSV. `--target` identifies the label column.
- Rows missing the target are removed. Feature rows and columns are otherwise kept.
- Put numeric-coded categorical feature names in the config's `categorical_columns`
  list. Object columns are converted to strings to handle mixed CSV value types.
- Each random seed creates one shared train/test split for every model.
- Classification splits are stratified when class counts and split sizes allow it.
- Baseline imputation, scaling, and one-hot encoding are fit inside a pipeline on the
  training rows. TabPFN receives the original DataFrame and applies its own pipeline.
- No hyperparameter search is run by default. Model defaults are kept, so baseline
  tuning can be added as a separate, documented experiment.

## Run

Install the repo with `python -m pip install -e ".[experiments]"`, place a dataset in
`datasets/raw/`, then use the root command shown in the main README. Override the YAML
dataset placeholder with `--data` and `--target`. A config can also be copied and
edited for a particular study.

The default version list is taken from `tabpfn.constants.ModelVersion` in the
included source: `v2`, `v2.5`, `v2.6`, `v3`, `v3.5`, and `v3.5-fast`. Default
baselines are dummy, linear, tree ensemble, and LightGBM estimators.

## Results

Raw results are stored in `experiments/results/raw_results/`, and processed results are stored in `experiments/results/processed_results/`.

## Artifacts

- `results/raw_results/`: one row per model, seed, and status; includes errors so a
  checkpoint access or size failure does not discard other model results.
- `results/processed_results/`: grouped mean and standard deviation tables.
- A JSON config snapshot and JSON run metadata sit next to each raw results CSV.
- `dataset_info.csv` is a metadata template; per-run profiles are stored in the
  run metadata JSON.

Fit time includes estimator construction and fitting. Classification prediction time
includes both hard-label and probability predictions when probabilities are available.
First-use weight downloads can affect the first fit. Peak process RAM is sampled at
50 ms intervals and does not include GPU memory. The raw file includes both the
process peak and its change from that model run's starting RSS; previously loaded
checkpoints or libraries can remain in process memory. Keep these conditions in mind
when interpreting resource results.

`experiments.evaluation.statistical_tests.paired_wilcoxon` compares two models over
matched seeds and applies a Holm correction across the comparisons it returns. The
tests remain exploratory when only a few random seeds are run; use a larger number of
prespecified seeds for formal comparisons.

Use `experiments.analysis.plot_results.plot_metric()` to save a metric comparison
chart under `results/plots/` from a raw run CSV.
