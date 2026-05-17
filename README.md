# Credit Risk PD Modeling and Scoring Pipeline

A machine learning project for estimating **Probability of Default (PD)** in a consumer credit risk setting.

This repository uses `data/credit_risk_data.csv` to run an end-to-end workflow to predict whether a borrower will experience serious delinquency within two years (`SeriousDlqin2yrs`).

The main final project notebook is:

- `Credit_Risk_PD_Modeling_and_Scoring_Pipeline.ipynb`

---

## Project Goals

- Build a binary classification model for default vs non-default risk.
- Provide a practical notebook-based baseline from exploration to evaluation.
- Export a reusable training and scoring pipeline for borrower-level PD, credit score, risk grade, and policy action.
- Keep the repository organized for GitHub collaboration and reproducibility.

## Key Results

The current reproducible training pipeline selects a calibrated `Random Forest` model and reports:

| Metric | Value |
|---|---:|
| Test ROC-AUC | 0.8655 |
| Test PR-AUC | 0.3776 |
| Brier score | 0.0493 |
| Selected PD threshold | 0.09 |
| Equivalent score cutoff | 550.8 |
| Precision at threshold | 0.2471 |
| Recall at threshold | 0.7264 |
| F1 at threshold | 0.3688 |
| Highest-risk decile default rate | about 37.0% |
| Highest-risk decile lift | about 5.53x |

Risk grades from the exported scoring pipeline are monotonic on the held-out test set:

| Grade | Risk meaning | Observed default rate | Avg score | Suggested action |
|---|---|---:|---:|---|
| E | Very high | 27.1% | 465.1 | Decline or enhanced verification |
| D | High | 7.1% | 569.9 | Manual review / risk-based pricing |
| C | Medium | 3.2% | 630.5 | Approve with review or lower limit |
| B | Low | 1.4% | 705.1 | Approve / standard terms |
| A | Very low | 0.5% | 781.4 | Auto approve / best terms |

---

## Repository Structure

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── models/
│   └── calibrated_pd_model.joblib
├── data/
│   ├── credit_risk_data.csv
│   └── credit_risk_data_dictionary.xls
├── outputs/
│   ├── cv_model_comparison.csv
│   ├── grade_summary.csv
│   ├── policy_threshold_table.csv
│   ├── run_summary.json
│   ├── scored_borrowers.csv
│   ├── scored_test_sample.csv
│   ├── scoring_examples.csv
│   ├── test_metrics.csv
│   ├── validation_model_comparison.csv
│   └── woe_iv_summary.csv
├── scripts/
│   ├── interactive_score.py
│   ├── score.py
│   └── train.py
├── src/
│   └── credit_risk/
│       ├── config.py
│       ├── evaluation.py
│       ├── models.py
│       ├── preprocessing.py
│       ├── scorecard.py
│       └── scoring.py
├── tests/
│   └── test_scoring.py
├── Credit_Risk_PD_Modeling_and_Scoring_Pipeline.ipynb
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Data

- `data/credit_risk_data.csv`: modeling dataset.
- `data/credit_risk_data_dictionary.xls`: variable definitions.
- Target column: `SeriousDlqin2yrs`.

> Do not commit personally identifiable information (PII) or confidential raw data to public repositories.

---

## Method Overview

The final notebook (`Credit_Risk_PD_Modeling_and_Scoring_Pipeline.ipynb`) is organized around the data science cycle and follows this flow:

1. Load the local CSV dataset and summarize the included data dictionary.
2. Perform compact EDA for missingness, imbalance, outliers, and default patterns.
3. Split train/validation/test before preprocessing to reduce leakage.
4. Fit preprocessing inside sklearn pipelines, including imputation, winsorization, clipping, missing indicators, and simple engineered features.
5. Add credit-risk-oriented engineered features, including delinquency totals, delinquency flags, log income, debt-per-income, and utilization flags.
6. Compare models suitable for tabular credit risk: Dummy baseline, balanced Logistic Regression, balanced Random Forest, and balanced HistGradientBoosting.
7. Select models using PR-AUC rather than accuracy because the default class is rare.
8. Calibrate predicted probabilities with isotonic calibration on a separate calibration/policy set.
9. Run cost sensitivity analysis and select an illustrative business-cost threshold.
10. Evaluate the final model on the held-out test set.
11. Review segment-level performance by age band, income missingness, and prior delinquency history.
12. Profile the highest-risk decile by age band, income missingness, and prior delinquency history.
13. Report ROC-AUC, PR-AUC, Brier score, precision, recall, F1, confusion matrix, lift by risk decile, feature importance, and SHAP explanations.
14. Convert calibrated PD values into a business-readable credit risk score using a log-odds scorecard mapping.
15. Assign A-E risk grades, summarize observed default rates by grade, and test the scoring mechanism with representative example PD values.
16. Export a reusable `src/` package plus `scripts/train.py` and `scripts/score.py` for repeatable training and batch scoring.
17. Produce WOE/IV diagnostics as a traditional scorecard-style interpretability check.
18. Summarize the final workflow in a model card and scoring policy card.

Current executed notebook results select `Random Forest` as the final model. On the held-out test set, the calibrated model reports approximately:

- ROC-AUC: `0.8655`
- PR-AUC: `0.3776`
- Brier score: `0.0493`
- Selected threshold: `0.09`
- Precision: `0.2471`
- Recall: `0.7264`
- F1: `0.3688`
- Highest-risk decile observed default rate: about `37.0%`
- Highest-risk decile lift: about `5.53x`
- Credit score range: `300` to `850`
- Selected PD threshold `0.09` maps to a score cutoff of about `550.8`
- A-E score grades show monotonically decreasing observed default rates from high-risk grade `E` to low-risk grade `A`

The notebook also includes a model card and scoring policy card covering intended use, data, selected model, calibration, threshold policy, score mapping, risk grades, test metrics, interpretability, and scope boundaries.

The training script auto-detects optional tabular boosters (`LightGBM`, `XGBoost`, `CatBoost`) if installed. In the checked environment these libraries are not installed, so the reproducible run uses sklearn models including `HistGradientBoosting` as the gradient-boosted baseline.

---

## Quick Start

### 1) Clone

```bash
git clone <your-repo-url>
cd credit-risk-pd-scoring-pipeline
```

### 2) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Launch notebook

```bash
jupyter notebook
```

Then open `Credit_Risk_PD_Modeling_and_Scoring_Pipeline.ipynb`.

### 4) Train and export artifacts

```bash
python scripts/train.py
```

This writes:

- `models/calibrated_pd_model.joblib`
- `outputs/cv_model_comparison.csv`
- `outputs/validation_model_comparison.csv`
- `outputs/test_metrics.csv`
- `outputs/grade_summary.csv`
- `outputs/scored_test_sample.csv`
- `outputs/scoring_examples.csv`
- `outputs/woe_iv_summary.csv`
- `outputs/run_summary.json`

By default the script uses `n_jobs=1` for portability. On a normal local machine, you can enable parallel grid search:

```bash
python scripts/train.py --n-jobs -1
```

### 5) Score borrowers

```bash
python scripts/score.py \
  --input data/credit_risk_data.csv \
  --output outputs/scored_borrowers.csv
```

Example scoring output:

| borrower_id | pd | credit_score | risk_grade | recommended_action |
|---|---:|---:|---|---|
| very_low_pd | 0.01 | 715.4 | B | Approve / standard terms |
| low_pd | 0.03 | 634.7 | C | Approve with review or lower limit |
| moderate_pd | 0.06 | 582.4 | D | Manual review / risk-based pricing |
| policy_threshold | 0.09 | 550.8 | D | Manual review / risk-based pricing |
| high_pd | 0.15 | 509.0 | E | Decline or enhanced verification |
| very_high_pd | 0.30 | 445.0 | E | Decline or enhanced verification |

### 6) Run tests

```bash
python -m pytest tests/test_scoring.py
```

### 7) Manual interactive scoring

After training has produced `models/calibrated_pd_model.joblib`, run:

```bash
python scripts/interactive_score.py
```

The script prompts for borrower fields such as utilization, age, delinquency counts, debt ratio, monthly income, open credit lines, real estate loans, and dependents. It then returns:

- probability of default
- 300-850 credit risk score
- A-E risk grade
- recommended action
- whether the borrower falls below the selected policy cutoff

---

## Resume Summary

Suggested resume wording:

> Built an end-to-end credit risk PD modeling and scoring pipeline on 150K borrower records, including leakage-safe preprocessing, stratified cross-validation, model comparison, isotonic probability calibration, cost-sensitive thresholding, SHAP interpretation, WOE/IV diagnostics, and a log-odds credit scoring mechanism with A-E risk grades. Achieved 0.865 ROC-AUC, 0.378 PR-AUC, 0.049 Brier score, 72.6% recall at the selected policy threshold, and 5.53x lift in the highest-risk decile.

---

## Suggested Next Improvements

- Replace the illustrative false-positive/false-negative costs with real credit economics.
- Install and benchmark `LightGBM`, `XGBoost`, or `CatBoost` for stronger tabular competition performance.
- Use nested validation for stricter model governance.
- Add bootstrap confidence intervals, calibration stability checks, and score drift monitoring.
- Extend fairness, drift, and reject-inference analysis if protected attributes, timestamps, and application-decision history become available.
- Expand the WOE/IV diagnostic into a full traditional logistic scorecard if a regulator-style scorecard is required.

---

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## License

This project is licensed under the [MIT License](LICENSE).
