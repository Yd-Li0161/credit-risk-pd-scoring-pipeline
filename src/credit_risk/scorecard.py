import numpy as np
import pandas as pd


def woe_iv_summary(X, y, max_bins=5, min_bin_size=0.03):
    """Create a compact WOE/IV diagnostic table for numeric scorecard review."""
    rows = []
    y = pd.Series(y).reset_index(drop=True)
    total_good = (y == 0).sum()
    total_bad = (y == 1).sum()
    eps = 0.5

    for feature in X.columns:
        s = pd.Series(X[feature]).reset_index(drop=True)
        if s.nunique(dropna=True) <= 1:
            continue

        try:
            bins = pd.qcut(s, q=max_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(s, bins=max_bins, duplicates="drop")

        labels = bins.astype("object")
        labels[s.isna()] = "Missing"
        table = pd.DataFrame({"bin": labels, "target": y})
        grouped = table.groupby("bin", observed=False).agg(
            borrowers=("target", "size"),
            bads=("target", "sum"),
        )
        grouped["goods"] = grouped["borrowers"] - grouped["bads"]
        grouped = grouped[grouped["borrowers"] / len(table) >= min_bin_size]
        if grouped.empty:
            continue

        grouped["bad_dist"] = (grouped["bads"] + eps) / (total_bad + eps * len(grouped))
        grouped["good_dist"] = (grouped["goods"] + eps) / (total_good + eps * len(grouped))
        grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
        grouped["iv_component"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]
        iv = grouped["iv_component"].sum()

        rows.append(
            {
                "feature": feature,
                "iv": iv,
                "bins_retained": len(grouped),
                "min_bin_borrowers": int(grouped["borrowers"].min()),
                "max_bin_borrowers": int(grouped["borrowers"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
