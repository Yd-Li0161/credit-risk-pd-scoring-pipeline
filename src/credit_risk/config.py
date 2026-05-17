RANDOM_STATE = 42
TARGET = "SeriousDlqin2yrs"
ID_COL = "Unnamed: 0"

SCORING_CONFIG = {
    "base_score": 600,
    "base_odds": 20,
    "pdo": 50,
    "min_score": 300,
    "max_score": 850,
}

RISK_GRADE_BINS = [-float("inf"), 540, 600, 660, 720, float("inf")]
RISK_GRADE_LABELS = ["E", "D", "C", "B", "A"]
RISK_GRADE_ACTIONS = {
    "A": "Auto approve / best terms",
    "B": "Approve / standard terms",
    "C": "Approve with review or lower limit",
    "D": "Manual review / risk-based pricing",
    "E": "Decline or enhanced verification",
}

