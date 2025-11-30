import numpy as np

from data.loader import BreastCancerDataLoader
from data.quality import DataQualityChecker
from dimensionality_reduction.lda_module import LDAReducer
from dimensionality_reduction.pca_module import PCAReducer
from eda.exploration import EDAExplorer
from evaluation.metrics import ModelEvaluator
from evaluation.reporting import ResultsReporter
from models.persistence import ModelPersistence
from models.trainer import ModelTrainer
from preprocessing.scaling import ScalerManager
from preprocessing.splitting import DataSplitter
from utils.config import Config, PathManager
from utils.plots import plot_confusion_matrix, plot_roc_curve
from xai.shap_analysis import SHAPAnalyzer


class PipelineOrchestrator:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.path_manager = PathManager(self.config)
        self.path_manager.ensure_directories_exist()

    def run_full_pipeline(self) -> None:
        # 1. Load data
        loader = BreastCancerDataLoader()
        X_df, y = loader.load_data()

        # 2. EDA
        eda = EDAExplorer()
        stats = eda.compute_descriptive_stats(X_df)
        print("\nDescriptive stats (first few rows):\n", stats.head())
        eda.plot_correlation(X_df, self.path_manager.get_plot_path("correlation_heatmap.png"))
        eda.plot_boxplots(X_df, self.path_manager.get_plot_path("boxplot_first10.png"))

        # 3 & 4. Split and scale
        splitter = DataSplitter()
        splits = splitter.split(
            X_df,
            y,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
        )

        X_train = splits["X_train"]
        X_val = splits["X_val"]
        X_test = splits["X_test"]
        y_train = splits["y_train"]
        y_val = splits["y_val"]
        y_test = splits["y_test"]

        scaler = ScalerManager(self.config.scaler_type)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled, X_test_scaled = scaler.transform_all_splits(X_val, X_test)

        representations: dict[str, dict[str, np.ndarray]] = {
            "raw": {
                "X_train": X_train_scaled,
                "X_val": X_val_scaled,
                "X_test": X_test_scaled,
            }
        }

        # 5. PCA
        pca_reducer = PCAReducer(self.config)
        pca_reducer.fit(X_train_scaled)
        pca_splits = pca_reducer.transform_all_splits(
            X_train_scaled, X_val_scaled, X_test_scaled
        )
        representations["pca"] = pca_splits
        pca_reducer.plot_explained_variance(
            self.path_manager.get_plot_path("pca_explained_variance.png")
        )
        pca_reducer.plot_2d_scatter(
            pca_splits["X_train"], y_train, self.path_manager.get_plot_path("pca_2d_scatter.png")
        )

        # 6. LDA
        lda_reducer = LDAReducer(self.config)
        lda_reducer.fit(X_train_scaled, y_train)
        lda_splits = lda_reducer.transform_all_splits(
            X_train_scaled, X_val_scaled, X_test_scaled
        )
        representations["lda"] = lda_splits
        lda_reducer.plot_projection(
            lda_splits["X_train"], y_train, self.path_manager.get_plot_path("lda_1d_hist.png")
        )

        # 6 & 7. Train models and evaluate on validation
        evaluator = ModelEvaluator()
        trainer = ModelTrainer(evaluator, path_manager=self.path_manager)
        validation_rows = trainer.train_and_validate_all(
            {rep: {"X_train": d["X_train"], "X_val": d["X_val"]} for rep, d in representations.items()},
            y_train,
            y_val,
            self.config.models_to_run,
            self.config.random_state,
        )

        reporter = ResultsReporter()
        reporter.save_validation_results(
            validation_rows,
            self.path_manager.get_results_csv_path("validation_results.csv"),
        )

        # 8. Best model on test set
        best_model_info = trainer.select_best_model(metric="roc_auc")
        best_model = best_model_info["model"]
        best_rep = best_model_info["representation"]
        X_test_best = representations[best_rep]["X_test"]

        if hasattr(best_model, "predict_proba"):
            y_test_proba = best_model.predict_proba(X_test_best)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_test_proba = best_model.decision_function(X_test_best)
        else:
            y_test_proba = None

        y_test_pred = best_model.predict(X_test_best)
        test_metrics = evaluator.compute_classification_metrics(
            y_test, y_test_pred, y_test_proba
        )
        cm = evaluator.confusion_matrix(y_test, y_test_pred)
        if y_test_proba is not None:
            roc_info = evaluator.roc_curve(y_test, y_test_proba)
        else:
            roc_info = None

        print("\nBest model on validation:", best_model_info["model_name"], "representation:", best_rep)
        print("Test metrics:", test_metrics)

        # Confusion matrix and ROC plots
        plot_confusion_matrix(
            cm,
            labels=["Class 0", "Class 1"],
            output_path=self.path_manager.get_plot_path("confusion_matrix.png"),
        )
        if roc_info is not None:
            plot_roc_curve(
                roc_info["fpr"],
                roc_info["tpr"],
                roc_info["auc"],
                self.path_manager.get_plot_path("roc_curve.png"),
            )

        reporter.save_test_results(
            {
                "representation": best_rep,
                "model_name": best_model_info["model_name"],
            },
            test_metrics,
            self.path_manager.get_results_csv_path("test_results.csv"),
        )

        # 9. SHAP analysis
        persistence = ModelPersistence(self.path_manager)
        # Save the best model separately under results/best_model
        persistence.save_best_model(best_model, best_rep, best_model_info["model_name"])

        shap_analyzer = SHAPAnalyzer(self.path_manager)
        best_per_rep = trainer.select_best_per_representation(metric="roc_auc")

        for rep_name, info in best_per_rep.items():
            model = info["model"]
            X_train_rep = representations[rep_name]["X_train"]
            X_test_rep = representations[rep_name]["X_test"]
            if rep_name == "raw":
                feature_names = list(X_df.columns)
            elif rep_name == "pca":
                n_comp = representations["pca"]["X_train"].shape[1]
                feature_names = [f"PC{i+1}" for i in range(n_comp)]
            else:  # lda
                n_comp = representations["lda"]["X_train"].shape[1]
                feature_names = [f"LD{i+1}" for i in range(n_comp)]

            shap_analyzer.explain_model(
                model,
                X_train_rep,
                X_test_rep,
                rep_name,
                info["model_name"],
                feature_names,
            )

        print("\nPipeline completed. Results saved under 'results/' and plots under 'results/plots/'.")
