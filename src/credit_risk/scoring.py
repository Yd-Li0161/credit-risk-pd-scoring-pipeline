import numpy as np
import pandas as pd

from .config import RISK_GRADE_ACTIONS, RISK_GRADE_BINS, RISK_GRADE_LABELS, SCORING_CONFIG


def score_from_pd(pd_values, config=SCORING_CONFIG):
    """Convert calibrated probability of default into bounded credit scores."""
    pd_array = np.asarray(pd_values, dtype=float)
    pd_array = np.clip(pd_array, 1e-6, 1 - 1e-6)
    factor = config["pdo"] / np.log(2)
    offset = config["base_score"] - factor * np.log(config["base_odds"])
    odds = (1 - pd_array) / pd_array
    score = offset + factor * np.log(odds)
    return np.clip(score, config["min_score"], config["max_score"])


def pd_from_score(scores, config=SCORING_CONFIG):
    """Invert the score mapping back to implied PD before clipping."""
    score_array = np.asarray(scores, dtype=float)
    factor = config["pdo"] / np.log(2)
    offset = config["base_score"] - factor * np.log(config["base_odds"])
    odds = np.exp((score_array - offset) / factor)
    return 1 / (1 + odds)


def assign_risk_grade(scores):
    return pd.cut(
        pd.Series(scores),
        bins=RISK_GRADE_BINS,
        labels=RISK_GRADE_LABELS,
        right=False,
        ordered=True,
    )


def build_scored_frame(ids, pd_scores, policy_pd_threshold=None):
    frame = pd.DataFrame({"pd": pd_scores})
    if ids is not None:
        frame.insert(0, "borrower_id", ids)
    frame["credit_score"] = score_from_pd(frame["pd"])
    frame["risk_grade"] = assign_risk_grade(frame["credit_score"]).astype(str).values
    frame["recommended_action"] = frame["risk_grade"].map(RISK_GRADE_ACTIONS)
    if policy_pd_threshold is not None:
        score_cutoff = float(score_from_pd([policy_pd_threshold])[0])
        frame["below_policy_cutoff"] = frame["credit_score"] <= score_cutoff
    return frame

