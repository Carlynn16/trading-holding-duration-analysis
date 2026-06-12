# Trading Holding Duration Analysis

**Quantifying how the risk of a losing trade grows with holding time — and turning that into an actionable exit rule for an automated forex trading operation.**

---

## Business problem

An automated forex trading operation runs strategies that open and close positions
continuously. Some positions are held too long and drift into losses. The objective of
this project is to answer one operational question:

> **From what holding duration onward does an open position become likely to turn into a loss — and when should it be closed to limit losses?**

Concretely, the analysis delivers (1) a holding-period → loss-probability table, (2) a
recommended maximum holding window, and (3) the trade characteristics that drive loss risk.

## Data

Two anonymized levels of trading data:

- **Trade level** — one row per executed trade (open/close timestamps, instrument, side,
  volume, prices, realized profit). Holding duration is derived as the time between open
  and close. This is the primary analysis dataset.
- **Strategy level** — one row per trading strategy, with summary performance and risk
  statistics (growth, Sharpe ratio, profit factor, win/loss counts, average holding time, …).

> **Data is not included in this repository.** The underlying trades are proprietary and
> are anonymized inside the pipeline (personal identifiers and strategy names are stripped).
> All code, tests, figures, and the report are fully reproducible once the source files are
> placed in the (git-ignored) `data/` directory.

## Approach

The analysis is built in three layers, from descriptive to inferential to predictive:

1. **Descriptive analysis** — distribution of holding durations and profits, and the
   relationship between holding duration and loss probability, broken down into holding-time
   bins (hours and days). Non-parametric tests (Spearman correlation, Kruskal–Wallis with
   Dunn post-hoc and multiple-testing correction) and effect sizes, since profit
   distributions are heavily skewed.

2. **Survival analysis** — the core of the project. Holding a trade is treated as a
   time-to-event problem: how long does a position "survive" before turning into a loss?
   - **Kaplan–Meier** survival curves (overall, by instrument, by trade side)
   - **Log-rank** tests for differences between groups
   - **Cox proportional-hazards** model with hazard ratios and a proportional-hazards
     assumption check

3. **Predictive modeling** — a comparison of classifiers (Logistic Regression, Random
   Forest, XGBoost, LightGBM) estimating loss probability from **entry-time information
   only** (no look-ahead leakage), evaluated with ROC/AUC, precision–recall, calibration,
   and SHAP feature attribution, with an honest discussion of the predictive ceiling.

## Key findings

<!-- TO COMPLETE as the analysis runs -->
- The probability of a losing trade rises monotonically with holding duration
  (≈ XX% for very short trades up to ≈ XX% beyond one day).
- Median profit turns negative around the **X–X day** holding range.
- Survival analysis: median time-to-loss is **XX**; hazard of loss is **X.X×** higher for
  [instrument/side] (Cox hazard ratios).
- Predictive model: best AUC **0.XX** using entry-time features — useful for ranking risk,
  with limits discussed in the report.

## Recommendation

<!-- TO COMPLETE -->
> Close positions after approximately **XX hours/days**, with instrument-specific
> adjustments and stop-loss rules detailed in the report.

## Repository structure

```
.
├── data/              # raw + processed data (git-ignored, not included)
├── figures/           # generated plots
├── notebooks/         # exploratory and analysis notebooks
├── scripts/           # report generation and pipeline entry points
├── src/               # data loading, preprocessing, analysis, modeling modules
├── tests/             # pytest unit tests
├── Statistical_Report.docx   # full written report
├── requirements.txt
└── pytest.ini
```

## Tech stack

Python · pandas · NumPy · SciPy · statsmodels · scikit-learn · XGBoost · LightGBM ·
lifelines (survival analysis) · SHAP · matplotlib · seaborn · python-docx · pytest

## Reproducing the analysis

```bash
pip install -r requirements.txt

# place the source CSVs in data/raw/ (not included), then:
python scripts/run_pipeline.py     # cleans data and writes processed parquet
pytest                             # run the test suite
```

## Report

The full analysis, figures, tables, and recommendations are in
**`Statistical_Report.docx`**, structured results-first: executive summary and exit
recommendation up front, supporting analysis after.
