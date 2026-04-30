# OR-Library Portfolio Optimization Experiments

This project runs multi-objective portfolio experiments on OR-Library datasets and provides:
- batch optimization runs (`weighted_sum`, `epsilon_constraint`, `nsga2`)
- sensitivity analysis for `mu` / `Sigma` perturbations
- paper-style figures
- portfolio composition (concentration/diversification) analysis

## 1) Python Requirements (Actual Imports)

Minimum:
- Python 3.8+

Standard-library imports used by this project:
- `argparse`
- `csv`
- `json`
- `math`
- `os`
- `random`
- `re`
- `statistics`
- `time`
- `typing`

Third-party imports required:
- `matplotlib` (`matplotlib.pyplot`)

Install third-party dependency:

```powershell
python -m pip install matplotlib
```

## 2) Data Layout

Place OR-Library files here:

```text
data/orlib/port1.txt
data/orlib/port2.txt
data/orlib/port3.txt
data/orlib/port4.txt
data/orlib/port5.txt
```

## 3) Main Scripts

- `run_orlib_batch.py`: run experiments across datasets/K values
- `run_orlib_sensitivity.py`: run scenario-based `mu`/`Sigma` sensitivity tests
- `analyze_sensitivity_results.py`: build comparison tables and sensitivity plots
- `make_paper_figures.py`: generate paper-ready figures from batch results
- `analyze_portfolio_composition.py`: analyze allocation concentration/dispersion
- `portfolio_methods.py`: optimization and data-loading backend

## 4) Quick Workflow

### Step A: Batch Run (all methods)

```powershell
python run_orlib_batch.py `
  --orlib-dir data/orlib `
  --datasets port1,port2,port3,port4,port5 `
  --k-values 2,4,8,16 `
  --run-nsga2 `
  --nsga2-k-values 2,4,8,16 `
  --output-dir results_orlib_allmethods
```

Main outputs:
- `results_orlib_allmethods/summary.csv`
- `results_orlib_allmethods/run_log.json`
- per-run folders like `port3_k8/` with:
  - `weighted_sum.csv`
  - `epsilon_constraint.csv`
  - `nsga2.csv`
  - `all_pareto.csv`
  - `pareto_points.csv`
  - `run_meta.json`

### Step B: Sensitivity Run

```powershell
python run_orlib_sensitivity.py
```

Default output:
- `results_orlib_allmethods/sensitivity_analysis/`
  - `sensitivity_summary.csv`
  - `sensitivity_log.json`
  - scenario folders and plots

### Step C: Sensitivity Comparison Tables + Plots

```powershell
python analyze_sensitivity_results.py `
  --summary-csv results_orlib_allmethods/sensitivity_analysis/sensitivity_summary.csv
```

Key outputs:
- `sensitivity_comparison_vs_base.csv`
- `sensitivity_method_aggregate_vs_base.csv`
- `fig10` to `fig13` (`.png` and `.pdf`)

### Step D: Paper Figures from Batch Results

```powershell
python make_paper_figures.py `
  --batch-dir results_orlib_allmethods `
  --output-dir results_orlib_allmethods/paper_figures
```

Outputs:
- `fig01_min_risk_vs_k`
- `fig02_max_return_vs_k`
- `fig03_runtime_by_method`
- `fig04_pareto_grid_k8`
- `fig05_pareto_grid_k16`

Each figure is exported as both `.png` and `.pdf`.

### Step E: Portfolio Concentration/Dispersion Analysis

```powershell
python analyze_portfolio_composition.py `
  --batch-dir results_orlib_allmethods `
  --output-dir results_orlib_allmethods/composition_analysis
```

Outputs:
- `composition_metrics.csv`
- `fig06_hhi_tradeoff` (`.png` / `.pdf`)
- `fig07_effective_n_by_k` (`.png` / `.pdf`)

## 5) Notes

- Use explicit `--batch-dir` and `--output-dir` to avoid mixing different experiment rounds.
- `make_paper_figures.py` expects a batch directory containing `summary.csv` and per-run folders.
- `analyze_portfolio_composition.py` expects run-folder names like `portX_kY`.
