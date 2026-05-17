import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """Leakage-safe preprocessing and feature engineering for credit risk data."""

    def __init__(self, add_engineered_features=True):
        self.add_engineered_features = add_engineered_features
        self.median_cols = ["MonthlyIncome", "NumberOfDependents"]
        self.winsor_cols = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
        self.upper_clip_values = {
            "NumberOfTime30-59DaysPastDueNotWorse": 10,
            "NumberOfTime60-89DaysPastDueNotWorse": 10,
            "NumberOfTimes90DaysLate": 10,
            "NumberRealEstateLoansOrLines": 5,
        }

    def fit(self, X, y=None):
        X = X.copy()
        self.feature_names_in_ = list(X.columns)
        self.medians_ = {col: X[col].median() for col in self.median_cols}
        self.winsor_bounds_ = {
            col: (X[col].quantile(0.01), X[col].quantile(0.99))
            for col in self.winsor_cols
        }
        transformed = self._transform_frame(X)
        self.feature_names_out_ = list(transformed.columns)
        return self

    def transform(self, X):
        return self._transform_frame(X.copy())

    def _transform_frame(self, X):
        X = X[self.feature_names_in_].copy()

        X["MonthlyIncome_missing"] = X["MonthlyIncome"].isna().astype(int)
        X["NumberOfDependents_missing"] = X["NumberOfDependents"].isna().astype(int)

        for col, median_value in self.medians_.items():
            X[col] = X[col].fillna(median_value)

        for col, (lower, upper) in self.winsor_bounds_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)

        for col, upper in self.upper_clip_values.items():
            X[col] = X[col].clip(upper=upper)

        if self.add_engineered_features:
            delinquency_cols = [
                "NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
            ]
            X["DelinquencyCountTotal"] = X[delinquency_cols].sum(axis=1)
            X["AnyDelinquency"] = (X["DelinquencyCountTotal"] > 0).astype(int)
            X["SevereDelinquency"] = (X["NumberOfTimes90DaysLate"] > 0).astype(int)
            X["LogMonthlyIncome"] = np.log1p(X["MonthlyIncome"].clip(lower=0))
            X["DebtPerIncome"] = X["DebtRatio"] * X["MonthlyIncome"]
            X["CreditPerProperty"] = X["NumberOfOpenCreditLinesAndLoans"] / (
                X["NumberRealEstateLoansOrLines"] + 1
            )
            X["UtilizationOverOne"] = (
                X["RevolvingUtilizationOfUnsecuredLines"] > 1
            ).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

