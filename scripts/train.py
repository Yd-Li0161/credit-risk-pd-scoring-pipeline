import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.config import ID_COL, RANDOM_STATE, TARGET
from credit_risk.evaluation import grade_summary, model_scores, threshold_table_from_scores
from credit_risk.models import candidate_models
from credit_risk.scorecard import woe_iv_summary
from credit_risk.scoring import build_scored_frame, score_from_pd


def split_data(df):
    model_df = df[df["age"] >= 18].copy()
    X = model_df.drop(columns=[ID_COL, TARGET])
    y = model_df[TARGET]

    X_train_valid_policy, X_test, y_train_valid_policy, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train_valid_policy,
        y_train_valid_policy,
        test_size=0.40,
        stratify=y_train_valid_policy,
        random_state=RANDOM_STATE,
    )
    X_valid, X_policy, y_valid, y_policy = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    return model_df, X_train, X_valid, X_policy, X_test, y_train, y_valid, y_policy, y_test


def fit_candidates(X_train, y_train, include_optional_boosters=True, n_jobs=1):
    best_models = {}
    rows = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, spec in candidate_models(include_optional_boosters=include_optional_boosters).items():
        if spec["params"]:
            search = GridSearchCV(
                estimator=spec["pipeline"],
                param_grid=spec["params"],
                scoring="average_precision",
                cv=cv,
                n_jobs=n_jobs,
            )
            search.fit(X_train, y_train)
            best_models[name] = search.best_estimator_
            cv_std = search.cv_results_["std_test_score"][search.best_index_]
            rows.append(
                {
                    "model": name,
                    "best_cv_pr_auc_mean": search.best_score_,
                    "best_cv_pr_auc_std": cv_std,
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                }
            )
        else:
            spec["pipeline"].fit(X_train, y_train)
            best_models[name] = spec["pipeline"]
            rows.append(
                {
                    "model": name,
                    "best_cv_pr_auc_mean": None,
                    "best_cv_pr_auc_std": None,
                    "best_params": "{}",
                }
            )
    return best_models, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and export the credit risk PD scoring pipeline.")
    parser.add_argument("--data", default="data/credit_risk_data.csv")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--fn-cost", type=float, default=10)
    parser.add_argument("--fp-cost", type=float, default=1)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV. Default is 1 for maximum portability.",
    )
    parser.add_argument(
        "--skip-optional-boosters",
        action="store_true",
        help="Skip LightGBM/XGBoost/CatBoost auto-detection even if installed.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.models_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    (
        model_df,
        X_train,
        X_valid,
        X_policy,
        X_test,
        y_train,
        y_valid,
        y_policy,
        y_test,
    ) = split_data(df)

    best_models, cv_results = fit_candidates(
        X_train,
        y_train,
        include_optional_boosters=not args.skip_optional_boosters,
        n_jobs=args.n_jobs,
    )

    validation_rows = []
    for name, model in best_models.items():
        validation_rows.append({"model": name, **model_scores(model, X_valid, y_valid)})
    validation_results = pd.DataFrame(validation_rows).sort_values("pr_auc", ascending=False)
    selected_model_name = validation_results.iloc[0]["model"]
    selected_model = best_models[selected_model_name]

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(selected_model),
        method="isotonic",
    )
    calibrated_model.fit(X_policy, y_policy)

    policy_pd = calibrated_model.predict_proba(X_policy)[:, 1]
    threshold_table = threshold_table_from_scores(
        y_policy,
        policy_pd,
        false_negative_cost=args.fn_cost,
        false_positive_cost=args.fp_cost,
    )
    selected_threshold = float(
        threshold_table.sort_values("expected_cost_per_1000", ascending=True).iloc[0]["threshold"]
    )

    test_pd = calibrated_model.predict_proba(X_test)[:, 1]
    test_metrics = pd.DataFrame(
        [
            {
                "selected_model": selected_model_name,
                "threshold": selected_threshold,
                "score_cutoff": float(score_from_pd([selected_threshold])[0]),
                **model_scores(calibrated_model, X_test, y_test, threshold=selected_threshold),
            }
        ]
    )

    scored_test = build_scored_frame(
        ids=X_test.index,
        pd_scores=test_pd,
        policy_pd_threshold=selected_threshold,
    )
    scored_test.insert(1, "actual_default", y_test.values)
    grade_results = grade_summary(scored_test, y_test)

    examples_pd = [0.01, 0.03, 0.06, selected_threshold, 0.15, 0.30]
    scoring_examples = build_scored_frame(
        ids=["very_low_pd", "low_pd", "moderate_pd", "policy_threshold", "high_pd", "very_high_pd"],
        pd_scores=examples_pd,
        policy_pd_threshold=selected_threshold,
    )

    woe_iv = woe_iv_summary(X_train, y_train)

    cv_results.to_csv(outputs_dir / "cv_model_comparison.csv", index=False)
    validation_results.to_csv(outputs_dir / "validation_model_comparison.csv", index=False)
    threshold_table.to_csv(outputs_dir / "policy_threshold_table.csv", index=False)
    test_metrics.to_csv(outputs_dir / "test_metrics.csv", index=False)
    grade_results.to_csv(outputs_dir / "grade_summary.csv", index=False)
    scored_test.head(1000).to_csv(outputs_dir / "scored_test_sample.csv", index=False)
    scoring_examples.to_csv(outputs_dir / "scoring_examples.csv", index=False)
    woe_iv.to_csv(outputs_dir / "woe_iv_summary.csv", index=False)

    bundle = {
        "model": calibrated_model,
        "selected_model_name": selected_model_name,
        "selected_threshold": selected_threshold,
        "feature_columns": list(X_train.columns),
        "target": TARGET,
        "id_col": ID_COL,
    }
    joblib.dump(bundle, models_dir / "calibrated_pd_model.joblib")

    run_summary = {
        "rows_after_age_filter": int(len(model_df)),
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "score_cutoff": float(score_from_pd([selected_threshold])[0]),
        "test_roc_auc": float(test_metrics.loc[0, "roc_auc"]),
        "test_pr_auc": float(test_metrics.loc[0, "pr_auc"]),
        "test_brier": float(test_metrics.loc[0, "brier"]),
        "test_precision": float(test_metrics.loc[0, "precision"]),
        "test_recall": float(test_metrics.loc[0, "recall"]),
        "test_f1": float(test_metrics.loc[0, "f1"]),
        "optional_boosters_used": [
            name for name in best_models if name in {"LightGBM", "XGBoost", "CatBoost"}
        ],
    }
    (outputs_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
