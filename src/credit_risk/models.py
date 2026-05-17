from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .preprocessing import CreditRiskPreprocessor


def candidate_models(include_optional_boosters=True):
    models = {
        "Dummy": {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    ("model", DummyClassifier(strategy="prior", random_state=RANDOM_STATE)),
                ]
            ),
            "params": {},
        },
        "Logistic Regression": {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="liblinear",
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__penalty": ["l1", "l2"],
                "model__C": [0.01, 0.1, 1.0],
            },
        },
        "Random Forest": {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=150,
                            class_weight="balanced_subsample",
                            n_jobs=1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__max_depth": [5, 10],
                "model__min_samples_leaf": [25, 75],
            },
        },
        "HistGradientBoosting": {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 0.1],
            },
        },
    }

    if include_optional_boosters:
        models.update(optional_booster_models())

    return models


def optional_booster_models():
    boosters = {}

    try:
        from lightgbm import LGBMClassifier

        boosters["LightGBM"] = {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    (
                        "model",
                        LGBMClassifier(
                            objective="binary",
                            n_estimators=300,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__learning_rate": [0.03, 0.05],
                "model__num_leaves": [15, 31],
                "model__min_child_samples": [50, 100],
            },
        }
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        boosters["XGBoost"] = {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="aucpr",
                            n_estimators=300,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            "params": {
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [3, 4],
                "model__min_child_weight": [20, 50],
            },
        }
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        boosters["CatBoost"] = {
            "pipeline": Pipeline(
                [
                    ("prep", CreditRiskPreprocessor()),
                    (
                        "model",
                        CatBoostClassifier(
                            loss_function="Logloss",
                            eval_metric="PRAUC",
                            iterations=300,
                            class_weights=[1, 10],
                            random_seed=RANDOM_STATE,
                            verbose=False,
                        ),
                    ),
                ]
            ),
            "params": {
                "model__learning_rate": [0.03, 0.05],
                "model__depth": [4, 6],
            },
        }
    except Exception:
        pass

    return boosters
