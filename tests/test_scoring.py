import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.preprocessing import CreditRiskPreprocessor
from credit_risk.scoring import assign_risk_grade, pd_from_score, score_from_pd


def test_score_decreases_as_pd_increases():
    pd_values = np.array([0.01, 0.03, 0.06, 0.09, 0.15, 0.30])
    scores = score_from_pd(pd_values)
    assert np.all(np.diff(scores) < 0)


def test_score_roundtrip_to_pd():
    pd_values = np.array([0.01, 0.03, 0.06, 0.09, 0.15, 0.30])
    scores = score_from_pd(pd_values)
    assert np.allclose(pd_from_score(scores), pd_values, atol=1e-10)


def test_representative_risk_grades():
    scores = score_from_pd([0.01, 0.03, 0.09, 0.30])
    grades = assign_risk_grade(scores).astype(str).tolist()
    assert grades == ["B", "C", "D", "E"]


def test_preprocessor_handles_missing_values_and_engineers_features():
    frame = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.45, 1.50, 0.05],
            "age": [45, 32, 70],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 2, 0],
            "DebtRatio": [0.35, 0.80, 0.10],
            "MonthlyIncome": [6000.0, np.nan, 3000.0],
            "NumberOfOpenCreditLinesAndLoans": [8, 4, 12],
            "NumberOfTimes90DaysLate": [0, 1, 0],
            "NumberRealEstateLoansOrLines": [1, 0, 2],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1, 0],
            "NumberOfDependents": [0.0, np.nan, 2.0],
        }
    )

    transformed = CreditRiskPreprocessor().fit_transform(frame)

    assert not transformed.isna().any().any()
    assert "MonthlyIncome_missing" in transformed.columns
    assert "NumberOfDependents_missing" in transformed.columns
    assert "DelinquencyCountTotal" in transformed.columns
    assert "AnyDelinquency" in transformed.columns
    assert transformed.loc[1, "MonthlyIncome_missing"] == 1
    assert transformed.loc[1, "NumberOfDependents_missing"] == 1
    assert transformed.loc[1, "AnyDelinquency"] == 1


def test_model_bundle_has_required_scoring_contract():
    bundle = joblib.load(PROJECT_ROOT / "models" / "calibrated_pd_model.joblib")

    assert {"model", "feature_columns", "selected_threshold", "target", "id_col"}.issubset(bundle)
    assert hasattr(bundle["model"], "predict_proba")
    assert len(bundle["feature_columns"]) == 10
    assert 0 < bundle["selected_threshold"] < 1


def test_batch_scoring_script_scores_small_sample(tmp_path):
    sample = pd.read_csv(PROJECT_ROOT / "data" / "credit_risk_data.csv").head(5)
    input_path = tmp_path / "borrowers.csv"
    output_path = tmp_path / "scored.csv"
    sample.to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    scored = pd.read_csv(output_path)
    assert "Wrote 5 scored borrowers" in result.stdout
    assert len(scored) == 5
    assert {"borrower_id", "pd", "credit_score", "risk_grade", "recommended_action"}.issubset(scored.columns)
    assert scored["pd"].between(0, 1).all()
    assert scored["credit_score"].between(300, 850).all()
    assert scored["risk_grade"].isin(["A", "B", "C", "D", "E"]).all()


def test_batch_scoring_script_fails_clearly_when_required_feature_missing(tmp_path):
    sample = pd.read_csv(PROJECT_ROOT / "data" / "credit_risk_data.csv").head(3)
    sample = sample.drop(columns=["DebtRatio"])
    input_path = tmp_path / "missing_feature.csv"
    output_path = tmp_path / "scored.csv"
    sample.to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "score.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "DebtRatio" in result.stderr
    assert not output_path.exists()

