# Three-person contribution plan

The remote repository will start with Part 1 only. The remaining two parts are
separate workstreams so each teammate can own and push a complete area.

## Part 1 — foundation (initial push)

- Official TabPFN implementation and its upstream licenses, notices, examples,
  documentation, and tests.
- Project metadata, dependency setup, experiment YAML files, and dataset/result folder
  layout.
- Dataset CSV loading, target validation, dataset metadata, and seeded shared splits.
- Explicit TabPFN checkpoint selection through the upstream version enum.

Relevant paths: `src/tabpfn/`, `pyproject.toml`, `experiments/configs/`,
`experiments/preprocessing/data_loading.py`,
`experiments/preprocessing/data_splitting.py`, and
`experiments/models/tabpfn_models.py`.

## Part 2 — model execution and baselines

One teammate owns the estimator layer and benchmark orchestration:

- Classification and regression baseline factories: dummy, linear, tree ensembles,
  and LightGBM.
- Baseline-only imputation, categorical encoding, and scaling fitted on each
  training split.
- A config-driven runner that compares every configured TabPFN version and baseline
  on the same seeded splits, records failures per model, and saves raw runs.
- Model settings and clear handling for checkpoints whose size limits do not fit a
  dataset.

Suggested paths: `experiments/models/baseline_models.py`,
`experiments/models/model_config.py`, and `experiments/run_experiments.py`.

## Part 3 — evaluation and reporting

The other teammate owns scoring, resource analysis, and research outputs:

- Classification and regression metrics, with undefined scores handled explicitly.
- Fit and prediction timing, process RAM tracking, and run metadata.
- Mean/standard-deviation summaries and paired statistical comparisons across seeds.
- Dataset/performance analysis, comparison plots, notebooks, and final result tables.

Suggested paths: `experiments/evaluation/`, `experiments/analysis/`,
`experiments/notebooks/`, and `experiments/results/`.

## Shared interface

Part 2 consumes the Part 1 dataset and split helpers, then calls the Part 3 metric and
resource interfaces. Agree on result column names before integrating so both parts
can be developed independently. Use one branch per part and merge through pull
requests to keep attribution visible.

The local full framework contains draft implementations for Parts 2 and 3. The
initial remote snapshot will omit those modules so the other two contributors can
own, refine, and push those workstreams.
