import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_correlation_heatmap(df: pd.DataFrame, output_path):
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_boxplots(df: pd.DataFrame, feature_names, output_path):
    # Adjust to use all columns of the provided DataFrame
    plt.figure(figsize=(12, 6))
    df.boxplot(rot=90)
    plt.title("Feature Boxplots")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_explained_variance(explained_variance_ratio, output_path):
    plt.figure(figsize=(8, 5))
    n_components = len(explained_variance_ratio)
    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.5, label="Individual")
    plt.step(
        range(1, n_components + 1),
        explained_variance_ratio.cumsum(),
        where="mid",
        label="Cumulative",
    )
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pca_2d_scatter(X_pca, y, output_path):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Scatter (PC1 vs PC2)")
    plt.colorbar(scatter, label="Class")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_lda_1d_projection(X_lda, y, output_path):
    plt.figure(figsize=(8, 5))
    # X_lda is 2D array with shape (n_samples, n_components_effective)
    comp1 = X_lda[:, 0]
    for label in np.unique(y):
        sns.kdeplot(comp1[y == label], label=f"Class {label}")
    plt.xlabel("LD1")
    plt.title("LDA 1D Projection Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(fpr, tpr, auc, output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
