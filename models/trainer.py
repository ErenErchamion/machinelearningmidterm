from typing import Dict, List

import numpy as np

from evaluation.metrics import ModelEvaluator
from models.registry import ModelRegistry
from models.persistence import ModelPersistence
from utils.config import PathManager, Config


class ModelTrainer:
    """Trains models on different data representations and evaluates on validation set."""

    def __init__(self, evaluator: ModelEvaluator, path_manager: PathManager | None = None):
        self.evaluator = evaluator
        self.results: List[Dict] = []
        self.persistence = ModelPersistence(path_manager) if path_manager is not None else None

    def train_and_validate_all(
        self,
        representations: Dict[str, Dict[str, np.ndarray]],
        y_train: np.ndarray,
        y_val: np.ndarray,
        model_names: List[str],
        random_state: int,
    ) -> List[Dict]:
        self.results = []
        for rep_name, data in representations.items():
            X_train = data["X_train"]
            X_val = data["X_val"]
            for model_name in model_names:
                try:
                    model = ModelRegistry.get_model(model_name, random_state)
                except ImportError:
                    # Skip XGBoost if not installed
                    continue
                model.fit(X_train, y_train)
                # Her eğitilen modeli results/models altına kaydet
                if self.persistence is not None:
                    self.persistence.save_model(model, rep_name, model_name)
                y_val_pred = model.predict(X_val)
                if hasattr(model, "predict_proba"):
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_val_proba = model.decision_function(X_val)
                else:
                    y_val_proba = None
                metrics = self.evaluator.compute_classification_metrics(
                    y_val, y_val_pred, y_val_proba
                )
                # Satırda kısa bir model adı tut, tam model nesnesini ayrıca sakla
                row = {
                    "representation": rep_name,
                    "model_name": model_name,
                    **metrics,
                }
                # Tam model nesnesi sadece bellek içinde tutulacak
                row_with_model = {**row, "model": model}
                self.results.append(row_with_model)
        return self.results

    def select_best_model(self, metric: str = "roc_auc") -> Dict:
        if not self.results:
            raise RuntimeError("No models have been trained.")

        def metric_value(row):
            value = row.get(metric)
            if value is None:
                return -1e9
            return value

        best = max(self.results, key=metric_value)
        return best

    def select_best_per_representation(self, metric: str = "roc_auc") -> Dict[str, Dict]:
        best_per_rep: Dict[str, Dict] = {}

        for row in self.results:
            rep = row["representation"]
            value = row.get(metric)
            if value is None:
                value = -1e9
            if rep not in best_per_rep or value > best_per_rep[rep].get(metric, -1e9):
                best_per_rep[rep] = row

        return best_per_rep
