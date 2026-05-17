import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import RISK_GRADE_LABELS


def model_scores(model, X_eval, y_eval, threshold=0.5):
    y_score = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_eval, y_score),
        "pr_auc": average_precision_score(y_eval, y_score),
        "brier": brier_score_loss(y_eval, y_score),
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall": recall_score(y_eval, y_pred, zero_division=0),
        "f1": f1_score(y_eval, y_pred, zero_division=0),
    }


def threshold_table_from_scores(
    y_true,
    y_score,
    false_negative_cost=10,
    false_positive_cost=1,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    rows = []
    n = len(y_true)
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        expected_cost = fp * false_positive_cost + fn * false_negative_cost
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "flagged_rate": y_pred.mean(),
                "expected_cost_per_1000": expected_cost / n * 1000,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def grade_summary(scored_frame, y_true):
    frame = scored_frame.copy()
    frame["y_true"] = np.asarray(y_true)
    frame["risk_grade"] = pd.Categorical(
        frame["risk_grade"],
        categories=RISK_GRADE_LABELS,
        ordered=True,
    )
    summary = (
        frame.groupby("risk_grade", observed=False)
        .agg(
            borrowers=("y_true", "size"),
            borrower_share=("y_true", lambda s: len(s) / len(frame)),
            defaults=("y_true", "sum"),
            observed_default_rate=("y_true", "mean"),
            avg_pd=("pd", "mean"),
            min_score=("credit_score", "min"),
            avg_score=("credit_score", "mean"),
            max_score=("credit_score", "max"),
        )
        .reset_index()
    )
    summary["lift_vs_portfolio"] = summary["observed_default_rate"] / frame["y_true"].mean()
    return summary
