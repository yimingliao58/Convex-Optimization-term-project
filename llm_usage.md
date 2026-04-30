# LLM Usage Disclosure (ECE 509 Term Project)

## 1. Tool / Model
- Tool/Platform: OpenAI Codex-style IDE assistant/Gemini
- Provider: OpenAI/Google
- Model: GPT-5.3-Codex/Gemini 3.1pro

## 2. Dates of Use
- 2026-04-25 to 2026-04-29

## 3. What It Was Used For
- Brain Storm
  - Confirm whether the project chosen by our group meets the requirements of the course project 
  - Identify any areas that can be improved.
- Code implementation support for:
  - Building and refining experiment scripts:
    - `portfolio_methods.py`
    - `run_orlib_batch.py`
    - `run_orlib_sensitivity.py`
  - Building post-processing and visualization scripts:
    - `analyze_sensitivity_results.py`
    - `make_paper_figures.py`
    - `analyze_portfolio_composition.py`
  - Debugging and robustness checks:
    - Environment/import checks and execution checks
    - Parser fix for OR-Library text files with UTF-8 BOM
    - Plotting robustness fixes (variable subplot count, missing file guards)
    - CSV output integrity checks
- Writing support for:
  - Drafting and updating project documentation:
    - `README.md`
    - `llm_usage.md`
  - Improving reproducibility text:
    - command examples
    - input/output description
    - file/folder structure explanations

## 4. Prompt / Transcript Log
- Full logs are provided as:
  - (A) chat link: (https://gemini.google.com/share/e79129356c52)
  - (B) local evidence files/screenshots:
    - `llm_screenshots/Codex/...`
- Representative prompts:
  1. "Implement batch experiments on OR-Library port1..port5 with K in {2,4,8,16}."
  2. "Check whether this script runs correctly in my target conda environment."
  3. "Generate sensitivity experiments for mu and Sigma and export summary CSV."
  4. "Create publication-ready figures and organize outputs for paper."
  5. "Analyze how concentrated/dispersed Pareto portfolios are from position weights."
  6. "Update README with reproducible commands and dependency requirements."

## 5. Outputs Materially Relied Upon
- Source code files materially relied upon:
  - `portfolio_methods.py`
  - `run_orlib_batch.py`
  - `run_orlib_sensitivity.py`
  - `analyze_sensitivity_results.py`
  - `make_paper_figures.py`
  - `analyze_portfolio_composition.py`
  - `README.md`
- Generated result artifacts materially relied upon:
  - Batch results:
    - `results_orlib_allmethods/summary.csv`
    - `results_orlib_allmethods/run_log.json`
    - per-run Pareto and method CSVs under `results_orlib_allmethods/port*_k*/`
  - Sensitivity results:
    - `results_orlib_batch_final/sensitivity_analysis/sensitivity_summary.csv`
    - `results_orlib_batch_final/sensitivity_analysis/sensitivity_log.json`
    - sensitivity figures/tables (`fig08`-`fig13`, comparison CSV files)
  - Composition results:
    - `results_orlib_allmethods_20260429/composition_analysis/composition_metrics.csv`
    - `fig06_hhi_tradeoff.*`
    - `fig07_effective_n_by_k.*`

## 6. Verification / Editing Performed by Students
- Students manually reviewed and edited all LLM-generated code and text before final use.
- Verification actions performed by students:
  - Syntax validation (`py_compile`) for modified scripts
  - CLI sanity checks (`--help`) before long runs
  - Smoke test with a small synthetic OR-Library-style file
  - Full-run verification on OR-Library `port1..port5` datasets
  - Result checks for CSV completeness and field validity
  - Figure generation checks (`.png` and `.pdf` existence and readability)
- Final decisions to keep/modify/reject LLM suggestions were made by students.

## 7. Responsibility Statement
We confirm that all final report text, code, figures, and technical claims were reviewed by us.
We take full responsibility for correctness, citations, and final submission quality.

## 8. External Code / Data Clarification
- External data: OR-Library `port1..port5` datasets (`data/orlib/`).
- External code copying: No direct copy from external repositories.
- LLM assistance was used for ideation, coding assistance, debugging, and documentation drafting, then verified/edited by students.
