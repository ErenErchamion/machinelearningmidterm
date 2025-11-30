from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
except ImportError:  # graceful degradation if xgboost not installed
    XGBClassifier = None


class ModelRegistry:
    """Factory for supported classification models."""

    @staticmethod
    def get_model(name: str, random_state: int) -> Any:
        name = name.lower()
        if name == "logistic_regression":
            return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        if name == "decision_tree":
            return DecisionTreeClassifier(random_state=random_state)
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=200, random_state=random_state)
        if name == "gaussian_nb":
            return GaussianNB()
        if name == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed")
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        raise ValueError(f"Unknown model name: {name}")

