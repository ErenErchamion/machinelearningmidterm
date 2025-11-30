from typing import Dict, Optional

import numpy as np
from sklearn import metrics


class ModelEvaluator:
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)
        roc_auc = None
        if y_proba is not None:
            roc_auc = metrics.roc_auc_score(y_true, y_proba)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return metrics.confusion_matrix(y_true, y_pred)

    def roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, np.ndarray]:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
        auc = metrics.auc(fpr, tpr)
        return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}

