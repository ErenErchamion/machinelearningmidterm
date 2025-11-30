from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from utils.config import PathManager


class ModelPersistence:
    """Handles saving and loading of trained models using joblib."""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

    def save_model(self, model: BaseEstimator, representation: str, model_name: str) -> Path:
        path = self.path_manager.get_model_path(representation, model_name)
        joblib.dump(model, path)
        return path

    def save_best_model(self, model: BaseEstimator, representation: str, model_name: str) -> Path:
        """Save the best model under results/best_model directory."""
        path = self.path_manager.get_best_model_path(representation, model_name)
        joblib.dump(model, path)
        return path

    def load_model(self, representation: str, model_name: str) -> BaseEstimator:
        path = self.path_manager.get_model_path(representation, model_name)
        return joblib.load(path)
