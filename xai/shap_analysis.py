from typing import List

import numpy as np
import shap

from utils.config import PathManager


class SHAPAnalyzer:
    """Runs SHAP analysis for trained models on different representations."""

    def __init__(self, path_manager: PathManager, background_size: int = 100):
        self.path_manager = path_manager
        self.background_size = background_size

    def _get_explainer(self, model, X_background: np.ndarray):
        # Prefer tree explainer for tree-based models
        model_name = model.__class__.__name__.lower()
        if "forest" in model_name or "xgb" in model_name or "tree" in model_name:
            return shap.TreeExplainer(model)
        if "logistic" in model_name or "linear" in model_name:
            return shap.LinearExplainer(model, X_background)
        # fallback
        return shap.KernelExplainer(model.predict_proba, X_background)

    def _select_shap_values(self, shap_values, X_explain: np.ndarray, n_features: int):
        """Normalize SHAP output shapes for binary classification.

        Handles cases where shap_values is a list, a 2D array, or a 3D array
        produced by some explainers, and removes any constant offset column.
        """

        # Case 1: list of arrays (e.g., TreeExplainer list per class)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                arr = np.array(shap_values[1])
            else:
                arr = np.array(shap_values[-1])
        else:
            arr = np.array(shap_values)

        # If 3D, reduce to 2D by selecting class dimension
        if arr.ndim == 3:
            if arr.shape[0] in (1, 2):
                idx = min(1, arr.shape[0] - 1)
                arr = arr[idx]
            elif arr.shape[1] in (1, 2):
                idx = min(1, arr.shape[1] - 1)
                arr = arr[:, idx, :]

        # Now arr should be 2D: (n_samples, n_features or n_features+1)
        if arr.ndim != 2:
            n_samples = X_explain.shape[0]
            arr = arr.reshape(n_samples, -1)

        # Some explainers append a constant offset column; if we have one extra feature,
        # drop the last column.
        if arr.shape[1] == n_features + 1:
            arr = arr[:, :n_features]

        # If for some reason we still mismatch, clip to min shared dimension
        if arr.shape[1] != n_features:
            k = min(arr.shape[1], n_features)
            arr = arr[:, :k]

        return arr

    def explain_model(
            self,
            model,
            X_train_repr: np.ndarray,
            X_test_repr: np.ndarray,
            representation: str,
            model_name: str,
            feature_names: List[str],
    ) -> None:
        # Subsample background and test for efficiency
        n_bg = min(self.background_size, X_train_repr.shape[0])
        X_background = X_train_repr[
            np.random.choice(X_train_repr.shape[0], n_bg, replace=False)
        ]

        explainer = self._get_explainer(model, X_background)

        n_samples = min(200, X_test_repr.shape[0])
        X_explain = X_test_repr[:n_samples]

        shap_values_raw = explainer.shap_values(X_explain)
        shap_values_to_plot = self._select_shap_values(
            shap_values_raw, X_explain, n_features=len(feature_names)
        )

        bar_path = self.path_manager.get_plot_path(
            f"{representation}_{model_name}_shap_bar.png"
        )
        summary_path = self.path_manager.get_plot_path(
            f"{representation}_{model_name}_shap_summary.png"
        )

        shap.summary_plot(
            shap_values_to_plot,
            X_explain,
            feature_names=feature_names[: shap_values_to_plot.shape[1]],
            plot_type="bar",
            show=False,
        )
        shap.plots._utils.plt.tight_layout()
        shap.plots._utils.plt.savefig(bar_path)
        shap.plots._utils.plt.close()

        shap.summary_plot(
            shap_values_to_plot,
            X_explain,
            feature_names=feature_names[: shap_values_to_plot.shape[1]],
            show=False,
        )
        shap.plots._utils.plt.tight_layout()
        shap.plots._utils.plt.savefig(summary_path)
        shap.plots._utils.plt.close()
