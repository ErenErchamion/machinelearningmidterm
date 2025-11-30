import pandas as pd
from pandas import DataFrame


class DataQualityChecker:
    """Performs basic data quality checks: missing values, dtypes, outliers."""

    def check_missing_values(self, X: pd.DataFrame, y: pd.Series) -> DataFrame:
        missing_X = X.isnull().sum()
        missing_y = pd.Series({"target": y.isnull().sum()})
        return pd.concat([missing_X, missing_y])

    def check_dtypes(self, X: pd.DataFrame) -> pd.Series:
        return X.dtypes

    def detect_outliers_zscore(self, X: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
        zscores = (X - X.mean()) / X.std(ddof=0)
        outlier_mask = (zscores.abs() > threshold)
        outlier_counts = outlier_mask.sum()
        return outlier_counts
