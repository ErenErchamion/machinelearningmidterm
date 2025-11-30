import pandas as pd
from sklearn.datasets import load_breast_cancer


class BreastCancerDataLoader:
    """Loads the Breast Cancer Wisconsin dataset as pandas DataFrame and Series."""

    def load_data(self):
        dataset = load_breast_cancer()
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target, name="target")
        return X, y
