from typing import Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.config import Config
from utils import plots


class LDAReducer:
    """LDA tabanlı boyut indirgeme bileşeni.

    - Config.lda_n_components ile hedef bileşen sayısı belirlenir (ödevde 3 isteniyor).
    - Ancak LDA'da teorik maksimum bileşen sayısı min(n_features, n_classes - 1)'dir.
      Breast Cancer veri seti 2 sınıflı olduğundan pratikte en fazla 1 bileşen üretilebilir.
    - Bu nedenle fit sırasında efektif bileşen sayısı dinamik olarak hesaplanır.
    """

    def __init__(self, config: Config):
        self.config = config
        self.lda: LinearDiscriminantAnalysis | None = None
        # Hedef bileşen sayısı (ödev tanımına uygun olarak 3)
        self.n_components_target: int = self.config.lda_n_components

    def fit(self, X_train_scaled: np.ndarray, y_train: np.ndarray) -> None:
        # Gerçek veri kısıtlarına göre maksimum LDA bileşen sayısını hesapla
        n_classes = len(np.unique(y_train))
        max_components = min(X_train_scaled.shape[1], n_classes - 1)
        # Efektif bileşen sayısı: hedef ile teorik maksimumun minimumu
        effective_components = max(1, min(self.n_components_target, max_components))

        self.lda = LinearDiscriminantAnalysis(n_components=effective_components)
        self.lda.fit(X_train_scaled, y_train)

    def transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.lda is None:
            raise RuntimeError("LDA henüz fit edilmedi. Önce `fit` çağrılmalı.")
        return self.lda.transform(X_scaled)

    def transform_all_splits(
        self,
        X_train_scaled: np.ndarray,
        X_val_scaled: np.ndarray,
        X_test_scaled: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Train/val/test split'lerini LDA uzayına projekte eder."""
        X_train_lda = self.transform(X_train_scaled)
        X_val_lda = self.transform(X_val_scaled)
        X_test_lda = self.transform(X_test_scaled)
        return {
            "X_train": X_train_lda,
            "X_val": X_val_lda,
            "X_test": X_test_lda,
        }

    def plot_projection(
        self,
        X_train_lda: np.ndarray,
        y_train: np.ndarray,
        output_path,
    ) -> None:
        """İlk bileşen(ler) üzerinde sınıf ayrımını görselleştirir ve kaydeder."""
        plots.plot_lda_1d_projection(X_lda=X_train_lda, y=y_train, output_path=output_path)
