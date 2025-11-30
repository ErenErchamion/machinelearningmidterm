from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Creates train/validation/test splits with stratification."""

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        val_size: float,
        random_state: int,
    ) -> Dict[str, np.ndarray]:
        # First split off test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # Validation size relative to remaining train_val portion
        val_size_relative = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_relative,
            random_state=random_state,
            stratify=y_train_val,
        )

        return {
            "X_train": X_train.values,
            "X_val": X_val.values,
            "X_test": X_test.values,
            "y_train": y_train.values,
            "y_val": y_val.values,
            "y_test": y_test.values,
        }
