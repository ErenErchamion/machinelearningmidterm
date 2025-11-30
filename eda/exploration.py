import pandas as pd

from utils.plots import plot_correlation_heatmap, plot_boxplots


class EDAExplorer:
    """Temel keşifsel veri analizi (EDA) ve görselleştirme işlemlerini yapar."""

    def compute_descriptive_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """Her özellik için mean, median, min-max, std, Q1-Q3 istatistiklerini hesaplar."""
        desc = X.describe(percentiles=[0.25, 0.5, 0.75]).T
        desc = desc.rename(columns={
            "25%": "Q1",
            "50%": "median",
            "75%": "Q3",
        })
        return desc[["mean", "median", "min", "max", "std", "Q1", "Q3"]]

    def plot_correlation(self, X: pd.DataFrame, output_path) -> None:
        """Korelasyon heatmap'ini çizer ve verilen path'e kaydeder."""
        plot_correlation_heatmap(X, output_path)

    def plot_boxplots(self, X: pd.DataFrame, output_path) -> None:
        """Tüm sayısal özellikler için boxplot'ları çizer ve kaydeder."""
        # plot_boxplots imzası (df, feature_names, output_path) şeklinde
        plot_boxplots(X, feature_names=X.columns.tolist(), output_path=output_path)
