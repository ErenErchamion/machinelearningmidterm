from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    scaler_type: str = "standard"  # standard | minmax | robust
    pca_variance_threshold: float = 0.95
    lda_n_components: int = 3
    models_to_run: List[str] = None
    results_dir: Path = Path("results")
    plots_dir: Path = Path("results/plots")
    models_dir: Path = Path("results/models")
    best_model_dir: Path = Path("results/best_model")

    def __post_init__(self):
        if self.models_to_run is None:
            self.models_to_run = [
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "xgboost",
                "gaussian_nb",
            ]


class PathManager:
    def __init__(self, config: Config):
        self.config = config

    def ensure_directories_exist(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        self.config.plots_dir.mkdir(parents=True, exist_ok=True)
        self.config.best_model_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, representation: str, model_name: str):
        return self.config.models_dir / f"{representation}_{model_name}.joblib"

    def get_best_model_path(self, representation: str, model_name: str):
        return self.config.best_model_dir / f"{representation}_{model_name}.joblib"

    def get_plot_path(self, filename: str):
        return self.config.plots_dir / filename

    def get_results_csv_path(self, filename: str):
        return self.config.results_dir / filename
