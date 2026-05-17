import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.scoring import build_scored_frame


FIELD_PROMPTS = {
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Revolving utilization of unsecured lines",
        "meaning": "Ratio of unsecured revolving credit currently used, such as credit card utilization.",
        "help": "Example: 0.45. Values above 1 mean utilization exceeds available unsecured credit.",
        "range": "typical 0-1; model accepts >1 for over-limit usage",
        "default": 0.45,
    },
    "age": {
        "label": "Age",
        "meaning": "Borrower's age in years.",
        "help": "Example: 45.",
        "range": "18-100",
        "default": 45,
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "30-59 days past due count",
        "meaning": "Number of times the borrower was 30-59 days past due, excluding worse delinquencies.",
        "help": "Example: 0, 1, or 2.",
        "range": "0-10; higher means more recent delinquency",
        "default": 0,
    },
    "DebtRatio": {
        "label": "Debt ratio",
        "meaning": "Monthly debt payments, alimony, and living costs divided by monthly gross income.",
        "help": "Example: 0.35.",
        "range": "typical 0-2; dataset contains larger outliers",
        "default": 0.35,
    },
    "MonthlyIncome": {
        "label": "Monthly income",
        "meaning": "Borrower's reported monthly income.",
        "help": "Example: 6000. Press Enter to leave missing.",
        "range": "0-20000 typical monthly USD; blank allowed",
        "default": 6000,
        "allow_missing": True,
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open credit lines and loans",
        "meaning": "Number of open installment loans and revolving credit lines.",
        "help": "Example: 8.",
        "range": "0-30 typical count",
        "default": 8,
    },
    "NumberOfTimes90DaysLate": {
        "label": "90+ days late count",
        "meaning": "Number of times the borrower was 90 or more days past due.",
        "help": "Example: 0, 1, or 2.",
        "range": "0-10; severe delinquency count",
        "default": 0,
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real estate loans or lines",
        "meaning": "Number of mortgage and real-estate-backed loans or credit lines.",
        "help": "Example: 1.",
        "range": "0-5 typical count",
        "default": 1,
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "60-89 days past due count",
        "meaning": "Number of times the borrower was 60-89 days past due, excluding worse delinquencies.",
        "help": "Example: 0, 1, or 2.",
        "range": "0-10; higher means more recent delinquency",
        "default": 0,
    },
    "NumberOfDependents": {
        "label": "Number of dependents",
        "meaning": "Number of dependents in the borrower's household, excluding the borrower.",
        "help": "Example: 0, 1, or 2. Press Enter to leave missing.",
        "range": "0-10 typical count; blank allowed",
        "default": 0,
        "allow_missing": True,
    },
}


def print_field_reference(feature_columns):
    print("\nInput field reference")
    print("The model uses the following borrower-level fields. Suggested ranges are practical guidance, not hard validation limits.\n")
    for number, feature in enumerate(feature_columns, start=1):
        prompt = FIELD_PROMPTS.get(
            feature,
            {
                "label": feature,
                "meaning": "Model input field.",
                "range": "numeric value",
                "default": 0,
                "allow_missing": False,
            },
        )
        missing_rule = "blank allowed = missing" if prompt.get("allow_missing", False) else "blank uses default"
        print(f"{number}. {prompt['label']}")
        print(f"   Meaning: {prompt.get('meaning', 'Model input field.')}")
        print(f"   Range:   {prompt.get('range', 'numeric value')}")
        print(f"   Default: {prompt['default']} ({missing_rule})")
    print()


def parse_value(raw_value, default, allow_missing=False):
    value = raw_value.strip()
    if value == "":
        if allow_missing:
            return None
        return default
    return float(value)


def collect_borrower_inputs(feature_columns):
    print("\nManual borrower scoring")
    print_field_reference(feature_columns)
    print("Press Enter to use the shown default. Blank is treated as missing only where noted.\n")
    values = {}

    for feature in feature_columns:
        prompt = FIELD_PROMPTS.get(
            feature,
            {"label": feature, "help": "", "default": 0, "allow_missing": False},
        )
        default = prompt["default"]
        allow_missing = prompt.get("allow_missing", False)

        while True:
            missing_note = "blank = missing" if allow_missing else f"blank = default {default}"
            range_note = prompt.get("range", "numeric value")
            raw = input(
                f"{prompt['label']} "
                f"(range: {range_note}; default: {default}; {missing_note}): "
            )
            try:
                values[feature] = parse_value(raw, default, allow_missing=allow_missing)
                break
            except ValueError:
                print(f"Invalid number. {prompt.get('help', '')}".strip())

    frame = pd.DataFrame([values])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Interactively enter borrower fields and score credit risk.")
    parser.add_argument("--model", default="models/calibrated_pd_model.joblib")
    parser.add_argument("--borrower-id", default="manual_input")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    feature_columns = bundle["feature_columns"]
    borrower = collect_borrower_inputs(feature_columns)

    pd_score = bundle["model"].predict_proba(borrower)[:, 1]
    scored = build_scored_frame(
        ids=[args.borrower_id],
        pd_scores=pd_score,
        policy_pd_threshold=bundle["selected_threshold"],
    )

    row = scored.iloc[0]
    print("\nCredit risk result")
    print(f"Borrower ID:        {row['borrower_id']}")
    print(f"Probability of default: {row['pd']:.4f}")
    print(f"Credit score:       {row['credit_score']:.1f}")
    print(f"Risk grade:         {row['risk_grade']}")
    print(f"Recommended action: {row['recommended_action']}")
    print(f"Below policy cutoff: {bool(row['below_policy_cutoff'])}")


if __name__ == "__main__":
    main()
