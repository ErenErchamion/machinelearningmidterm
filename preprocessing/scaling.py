from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ScalerManager:
    def __init__(self, scaler_type: str = "standard"):
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def transform_all_splits(
        self, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(X_val), self.transform(X_test)

