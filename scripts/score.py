import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_risk.config import ID_COL, TARGET
from credit_risk.scoring import build_scored_frame


def main():
    parser = argparse.ArgumentParser(description="Score borrowers with the trained credit risk model.")
    parser.add_argument("--input", required=True, help="Input CSV with the model feature columns.")
    parser.add_argument("--model", default="models/calibrated_pd_model.joblib")
    parser.add_argument("--output", default="outputs/scored_borrowers.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.input)
    ids = df[ID_COL] if ID_COL in df.columns else df.index
    feature_columns = bundle["feature_columns"]
    X = df.drop(columns=[TARGET], errors="ignore")
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    missing_columns = [col for col in feature_columns if col not in X.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise SystemExit(f"Input is missing required feature columns: {missing}")

    X = X[feature_columns]

    pd_scores = bundle["model"].predict_proba(X)[:, 1]
    scored = build_scored_frame(
        ids=ids,
        pd_scores=pd_scores,
        policy_pd_threshold=bundle["selected_threshold"],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Wrote {len(scored):,} scored borrowers to {output_path}")


if __name__ == "__main__":
    main()
