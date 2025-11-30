from typing import Dict, Any

import numpy as np
from sklearn.decomposition import PCA

from utils.config import Config
from utils import plots


class PCAReducer:
    """PCA tabanlı boyut indirgeme bileşeni.

    - Tüm bileşenlerle PCA fit eder.
    - Config içindeki `pca_variance_threshold` değerine göre
      yeterli açıklanan varyansa ulaşmak için gereken bileşen sayısını seçer.
    - Seçilen bileşen sayısı ile yeniden PCA fit eder ve transform işlemlerini yapar.
    - İsteğe bağlı olarak explained variance grafiğini `results/plots` altına kaydeder.
    """

    def __init__(self, config: Config):
        self.config = config
        self.pca_full: PCA | None = None
        self.pca: PCA | None = None
        self.n_components_selected: int | None = None

    def fit(self, X_train_scaled: np.ndarray, plot: bool = True) -> None:
        """PCA'yı fit eder ve bileşen sayısını seçer."""
        # Tüm bileşenlerle PCA fit
        self.pca_full = PCA(random_state=self.config.random_state)
        self.pca_full.fit(X_train_scaled)

        explained = self.pca_full.explained_variance_ratio_
        cumulative = explained.cumsum()

        # Eşik değerini geçen ilk bileşen sayısını seç
        self.n_components_selected = int(
            np.argmax(cumulative >= self.config.pca_variance_threshold) + 1
        )

        # Seçilen bileşen sayısı ile PCA
        self.pca = PCA(
            n_components=self.n_components_selected,
            random_state=self.config.random_state,
        )
        self.pca.fit(X_train_scaled)

        if plot:
            self._plot_explained_variance(explained, cumulative)

    def transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Fit edilmiş PCA ile veriyi dönüştürür."""
        if self.pca is None:
            raise RuntimeError("PCA henüz fit edilmedi. Önce `fit` çağrılmalı.")
        return self.pca.transform(X_scaled)

    def transform_all_splits(
        self,
        X_train_scaled: np.ndarray,
        X_val_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Train/val/test split'lerini PCA uzayına projekte eder."""
        X_train_pca = self.transform(X_train_scaled)
        X_val_pca = self.transform(X_val_scaled)
        X_test_pca = self.transform(X_test_scaled)
        return {
            "X_train": X_train_pca,
            "X_val": X_val_pca,
            "X_test": X_test_pca,
        }

    def fit_transform(self, X_train_scaled: np.ndarray, plot: bool = True) -> np.ndarray:
        """Hem fit eder hem de dönüştürülmüş veriyi döner."""
        self.fit(X_train_scaled, plot=plot)
        return self.transform(X_train_scaled)

    def get_summary(self) -> Dict[str, Any]:
        """Rapor için PCA özetini döner."""
        if self.pca_full is None or self.n_components_selected is None:
            raise RuntimeError("PCA henüz fit edilmedi. Önce `fit` çağrılmalı.")

        explained = self.pca_full.explained_variance_ratio_
        cumulative = explained.cumsum()

        return {
            "n_components_total": int(len(explained)),
            "n_components_selected": int(self.n_components_selected),
            "variance_threshold": float(self.config.pca_variance_threshold),
            "explained_variance_ratio": explained.tolist(),
            "cumulative_explained_variance_ratio": cumulative.tolist(),
        }

    def _plot_explained_variance(
        self,
        explained: np.ndarray,
        cumulative: np.ndarray,
    ) -> None:
        """Explained variance grafiğini plots_dir altına kaydeder."""
        output_path = self.config.plots_dir / "pca_explained_variance.png"
        plots.plot_pca_explained_variance(
            explained_variance_ratio=explained,
            output_path=output_path,
        )

    def plot_explained_variance(self, output_path) -> None:
        """Orchestrator ile uyum için, explained variance grafiğini verilen path'e kaydeder."""
        if self.pca_full is None:
            raise RuntimeError("PCA henüz fit edilmedi. Önce `fit` çağrılmalı.")
        explained = self.pca_full.explained_variance_ratio_
        plots.plot_pca_explained_variance(explained_variance_ratio=explained, output_path=output_path)

    def plot_2d_scatter(self, X_train_pca: np.ndarray, y_train: np.ndarray, output_path) -> None:
        """İlk iki PCA bileşeni ile sınıf ayrışmasını 2D scatter plot olarak çizer."""
        plots.plot_pca_2d_scatter(X_pca=X_train_pca, y=y_train, output_path=output_path)
