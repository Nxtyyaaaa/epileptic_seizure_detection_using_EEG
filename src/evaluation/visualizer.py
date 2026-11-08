import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix


class ResultVisualizer:
    """Generates publication-quality figures and plots for seizure detection evaluation."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Clinical styling
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png",
    ):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Ictal (0)", "Ictal (1)"],
            yticklabels=["Non-Ictal (0)", "Ictal (1)"],
            cbar=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_pr_curves_comparison(
        self,
        experiment_results: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
        title: str = "Precision-Recall Curves Across Balancing Strategies",
        filename: str = "pr_curves_comparison.png",
    ):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        for name, (y_true, y_prob) in experiment_results.items():
            p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            precision, recall, _ = precision_recall_curve(y_true, p)
            ax.plot(recall, precision, lw=2, label=name)

        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_balancing_comparison_bar(
        self,
        df_results: pd.DataFrame,
        metrics: List[str] = ["sensitivity", "precision", "f1", "pr_auc"],
        title: str = "Comparison of Balancing Strategies (E0 - E6)",
        filename: str = "balancing_comparison_bar.png",
    ):
        """
        Plots a clustered bar chart comparing E0 through E6 across chosen metrics.
        """
        df_plot = df_results.copy()
        if "strategy" not in df_plot.columns:
            df_plot["strategy"] = df_plot.index

        melted = pd.melt(
            df_plot,
            id_vars=["strategy"],
            value_vars=[m for m in metrics if m in df_plot.columns],
            var_name="Metric",
            value_name="Score",
        )

        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        sns.barplot(data=melted, x="strategy", y="Score", hue="Metric", ax=ax, palette="Set2")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Balancing Strategy", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc="lower right")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        return save_path
