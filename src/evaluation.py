"""
evaluation.py - Model Evaluation and Visualization
====================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Comprehensive model evaluation with all metrics,
             confusion matrix, ROC curve, precision-recall curve,
             cross-validation, and model comparison plots.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils import CONFIG, FIGURES_DIR, logger, timer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT STYLING
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0E1117"
CARD_BG = "#1A1F2E"
ACCENT = "#00D4FF"
DANGER = "#FF6B6B"
SUCCESS = "#4ECDC4"
WARNING = "#FFD700"
TEXT = "#FAFAFA"
GRID = "#2A2F3E"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ModelEvaluator:
    """
    Comprehensive evaluation suite for churn prediction models.

    Features:
        - Full classification metrics
        - Confusion matrix visualization
        - ROC-AUC curve
        - Precision-Recall curve
        - Cross-validation analysis
        - Model comparison bar charts

    Attributes:
        model: Fitted sklearn estimator.
        model_name (str): Display name for the model.
        y_pred (np.ndarray): Test predictions.
        y_prob (np.ndarray): Predicted probabilities.
    """

    def __init__(self, model: Any, model_name: str = "Model") -> None:
        """
        Initialize evaluator with a fitted model.

        Args:
            model: Fitted sklearn estimator.
            model_name: Name for display in plots.
        """
        self.model = model
        self.model_name = model_name
        self.y_pred: Optional[np.ndarray] = None
        self.y_prob: Optional[np.ndarray] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_test: Optional[pd.Series] = None
        logger.info(f"ModelEvaluator initialized for: {model_name}")

    # ─────────────────────────────────────────────
    # GENERATE PREDICTIONS
    # ─────────────────────────────────────────────
    def predict(self,
                X_test: pd.DataFrame,
                y_test: pd.Series,
                threshold: float = 0.5) -> None:
        """
        Generate predictions and probabilities for test set.

        Args:
            X_test: Test feature matrix.
            y_test: True test labels.
            threshold: Classification threshold for predictions.
        """
        self._X_test = X_test
        self._y_test = y_test

        if hasattr(self.model, "predict_proba"):
            self.y_prob = self.model.predict_proba(X_test)[:, 1]
            self.y_pred = (self.y_prob >= threshold).astype(int)
        else:
            self.y_pred = self.model.predict(X_test)
            self.y_prob = self.y_pred.astype(float)

    # ─────────────────────────────────────────────
    # METRICS REPORT
    # ─────────────────────────────────────────────
    def get_metrics(self) -> Dict:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary with all metric values.
        """
        if self.y_pred is None:
            raise RuntimeError("Call predict() first.")

        y_test = self._y_test
        y_pred = self.y_pred
        y_prob = self.y_prob

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        avg_precision = (
            average_precision_score(y_test, y_prob)
            if y_prob is not None else np.nan
        )

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics = {
            "Model": self.model_name,
            "Accuracy": round(accuracy * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall (Sensitivity)": round(recall * 100, 2),
            "Specificity": round(specificity * 100, 2),
            "F1 Score": round(f1 * 100, 2),
            "ROC-AUC": round(roc_auc * 100, 2),
            "Avg Precision": round(avg_precision * 100, 2),
            "True Positives": int(tp),
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
        }

        logger.info(f"\n📊 {self.model_name} Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:<30} {v:.2f}%")
            else:
                logger.info(f"  {k:<30} {v}")

        return metrics

    # ─────────────────────────────────────────────
    # CONFUSION MATRIX PLOT
    # ─────────────────────────────────────────────
    @timer
    def plot_confusion_matrix(self,
                              save: bool = True,
                              filename: str = "confusion_matrix.png") -> plt.Figure:
        """
        Plot a styled confusion matrix heatmap.

        Args:
            save: Whether to save the figure.
            filename: Output filename.

        Returns:
            matplotlib Figure object.
        """
        if self.y_pred is None:
            raise RuntimeError("Call predict() first.")

        cm = confusion_matrix(self._y_test, self.y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(DARK_BG)

        # Custom colors
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            cm, annot=False, cmap="Blues", ax=ax,
            linewidths=2, linecolor=DARK_BG,
            cbar_kws={"shrink": 0.8}
        )

        # Annotate with both count and percentage
        labels = [
            ["TN\n{:,}\n({:.1f}%)".format(cm[0, 0], cm_pct[0, 0]),
             "FP\n{:,}\n({:.1f}%)".format(cm[0, 1], cm_pct[0, 1])],
            ["FN\n{:,}\n({:.1f}%)".format(cm[1, 0], cm_pct[1, 0]),
             "TP\n{:,}\n({:.1f}%)".format(cm[1, 1], cm_pct[1, 1])],
        ]
        colors_text = [[SUCCESS, DANGER], [WARNING, ACCENT]]
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.5, labels[i][j],
                        ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color=colors_text[i][j])

        ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
        ax.set_ylabel("True Label", fontsize=13, labelpad=10)
        ax.set_xticklabels(["Not Churned", "Churned"], fontsize=11)
        ax.set_yticklabels(["Not Churned", "Churned"],
                           fontsize=11, rotation=0)
        ax.set_title(f"Confusion Matrix — {self.model_name}",
                     fontsize=15, fontweight="bold", pad=15, color=TEXT)

        plt.tight_layout()
        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ Confusion matrix saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # ROC CURVE
    # ─────────────────────────────────────────────
    @timer
    def plot_roc_curve(self,
                       all_models: Optional[Dict[str, Any]] = None,
                       X_test: Optional[pd.DataFrame] = None,
                       y_test: Optional[pd.Series] = None,
                       save: bool = True,
                       filename: str = "roc_curve.png") -> plt.Figure:
        """
        Plot ROC curve for this model and optionally all compared models.

        Args:
            all_models: Dict of name → fitted model to plot together.
            X_test: Test features (needed if plotting multiple models).
            y_test: Test labels.
            save: Whether to save figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(DARK_BG)

        colors = CONFIG["color_palette"]

        if all_models and X_test is not None and y_test is not None:
            # Plot all models
            for idx, (name, mdl) in enumerate(all_models.items()):
                if hasattr(mdl, "predict_proba"):
                    prob = mdl.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, prob)
                    auc = roc_auc_score(y_test, prob)
                    lw = 3 if name == self.model_name else 1.5
                    alpha = 1.0 if name == self.model_name else 0.65
                    label = f"{name} (AUC={auc*100:.1f}%)"
                    if name == self.model_name:
                        label += " ⭐"
                    ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                            lw=lw, alpha=alpha, label=label)
        else:
            # Plot single model
            fpr, tpr, _ = roc_curve(self._y_test, self.y_prob)
            auc = roc_auc_score(self._y_test, self.y_prob)
            ax.plot(fpr, tpr, color=ACCENT, lw=3,
                    label=f"{self.model_name} (AUC={auc*100:.1f}%)")

        # Random classifier baseline
        ax.plot([0, 1], [0, 1], "w--", lw=1.5, alpha=0.5,
                label="Random Classifier (AUC=50%)")

        ax.fill_between(fpr if all_models is None else [0, 1],
                        tpr if all_models is None else [0, 0],
                        alpha=0.05, color=ACCENT)

        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title("ROC Curve — Model Comparison",
                     fontsize=15, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=10,
                  facecolor=CARD_BG, edgecolor=GRID)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        plt.tight_layout()
        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ ROC curve saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # PRECISION-RECALL CURVE
    # ─────────────────────────────────────────────
    @timer
    def plot_precision_recall_curve(self,
                                    save: bool = True,
                                    filename: str = "pr_curve.png") -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            save: Whether to save the figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        if self.y_prob is None:
            raise RuntimeError("Probabilities not available.")

        precision, recall, thresholds = precision_recall_curve(
            self._y_test, self.y_prob
        )
        avg_prec = average_precision_score(self._y_test, self.y_prob)

        # Find optimal threshold (max F1)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
        optimal_f1 = f1_scores[optimal_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(DARK_BG)

        # PR Curve
        ax1.plot(recall, precision, color=ACCENT, lw=2.5)
        ax1.fill_between(recall, precision, alpha=0.15, color=ACCENT)
        ax1.axhline(y=self._y_test.mean(), color=DANGER,
                    linestyle="--", lw=1.5,
                    label=f"Baseline (Churn Rate = {self._y_test.mean()*100:.1f}%)")
        ax1.scatter(recall[optimal_idx], precision[optimal_idx],
                    s=150, color=WARNING, zorder=5,
                    label=f"Optimal (F1={optimal_f1*100:.1f}%)")
        ax1.set_xlabel("Recall", fontsize=12)
        ax1.set_ylabel("Precision", fontsize=12)
        ax1.set_title(f"Precision-Recall Curve\n(AP={avg_prec*100:.1f}%)",
                      fontsize=13, fontweight="bold")
        ax1.legend(facecolor=CARD_BG, edgecolor=GRID, fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])

        # Threshold Analysis
        if len(thresholds) > 1:
            ax2.plot(thresholds, precision[:-1], color=SUCCESS,
                     lw=2, label="Precision")
            ax2.plot(thresholds, recall[:-1], color=DANGER,
                     lw=2, label="Recall")
            ax2.plot(thresholds, f1_scores[:-1], color=WARNING,
                     lw=2.5, label="F1 Score")
            ax2.axvline(x=optimal_threshold, color=ACCENT,
                        linestyle="--", lw=1.5,
                        label=f"Optimal Threshold={optimal_threshold:.2f}")
            ax2.set_xlabel("Decision Threshold", fontsize=12)
            ax2.set_ylabel("Score", fontsize=12)
            ax2.set_title("Precision / Recall / F1 vs Threshold",
                          fontsize=13, fontweight="bold")
            ax2.legend(facecolor=CARD_BG, edgecolor=GRID, fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f"Precision-Recall Analysis — {self.model_name}",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ PR curve saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # MODEL COMPARISON PLOT
    # ─────────────────────────────────────────────
    @staticmethod
    def plot_model_comparison(results_df: pd.DataFrame,
                              save: bool = True,
                              filename: str = "model_comparison.png") -> plt.Figure:
        """
        Create a grouped bar chart comparing all models across metrics.

        Args:
            results_df: DataFrame from ModelTrainer.train_all().
            save: Whether to save figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        df_plot = results_df[["Model"] + metrics].copy()
        for col in metrics:
            df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.patch.set_facecolor(DARK_BG)
        fig.suptitle("Model Performance Comparison",
                     fontsize=18, fontweight="bold", y=1.01)

        colors = CONFIG["color_palette"]

        # Plot 1: All metrics bar chart
        ax = axes[0]
        x = np.arange(len(df_plot))
        width = 0.15
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * width, df_plot[metric],
                          width=width, label=metric,
                          color=colors[i], alpha=0.85,
                          edgecolor=DARK_BG, linewidth=0.5)

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_plot["Model"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("All Metrics by Model", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9,
                  facecolor=CARD_BG, edgecolor=GRID)
        ax.set_ylim([0, 105])
        ax.grid(True, axis="y", alpha=0.3)

        # Plot 2: F1 Score ranking horizontal bar
        ax2 = axes[1]
        df_sorted = df_plot.sort_values("F1 Score", ascending=True)
        bar_colors = [SUCCESS if m == df_plot.loc[df_plot["F1 Score"].idxmax(),
                                                   "Model"]
                      else ACCENT for m in df_sorted["Model"]]
        bars = ax2.barh(df_sorted["Model"], df_sorted["F1 Score"],
                        color=bar_colors, alpha=0.85,
                        edgecolor=DARK_BG, linewidth=0.5)

        for bar, val in zip(bars, df_sorted["F1 Score"]):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", fontsize=10,
                     fontweight="bold", color=TEXT)

        ax2.set_xlabel("F1 Score (%)", fontsize=12)
        ax2.set_title("F1 Score Ranking (Best Model Highlighted)",
                      fontsize=13, fontweight="bold")
        ax2.set_xlim([0, 110])
        ax2.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ Model comparison saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # CROSS VALIDATION
    # ─────────────────────────────────────────────
    @timer
    def cross_validate(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       scoring: str = "f1",
                       cv: int = 5) -> Dict:
        """
        Run stratified k-fold cross validation.

        Args:
            X_train: Training features.
            y_train: Training target.
            scoring: Sklearn scoring metric string.
            cv: Number of folds.

        Returns:
            Dictionary with CV scores statistics.
        """
        kfold = StratifiedKFold(
            n_splits=cv, shuffle=True,
            random_state=CONFIG["random_state"]
        )
        scores = cross_val_score(
            self.model, X_train, y_train,
            cv=kfold, scoring=scoring, n_jobs=-1
        )

        result = {
            "scores": scores.tolist(),
            "mean": round(scores.mean() * 100, 2),
            "std": round(scores.std() * 100, 2),
            "min": round(scores.min() * 100, 2),
            "max": round(scores.max() * 100, 2),
        }
        logger.info(f"Cross-Validation ({cv}-fold {scoring}): "
                    f"{result['mean']:.2f}% ± {result['std']:.2f}%")
        return result

    # ─────────────────────────────────────────────
    # FULL EVALUATION REPORT
    # ─────────────────────────────────────────────
    @timer
    def full_evaluation_report(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        all_models: Optional[Dict] = None,
        save_plots: bool = True
    ) -> Dict:
        """
        Run the complete evaluation suite and return all results.

        Args:
            X_train, X_test: Feature splits.
            y_train, y_test: Target splits.
            all_models: All trained models for ROC comparison.
            save_plots: Whether to save all plots.

        Returns:
            Complete evaluation report dictionary.
        """
        logger.info("=" * 60)
        logger.info(f"STEP 5: Evaluating {self.model_name}")
        logger.info("=" * 60)

        self.predict(X_test, y_test)

        report = {
            "metrics": self.get_metrics(),
            "classification_report": classification_report(
                y_test, self.y_pred,
                target_names=["Not Churned", "Churned"]
            ),
            "cv_results": self.cross_validate(X_train, y_train),
        }

        # Generate plots
        self.plot_confusion_matrix(save=save_plots)
        self.plot_roc_curve(
            all_models=all_models,
            X_test=X_test,
            y_test=y_test,
            save=save_plots
        )
        self.plot_precision_recall_curve(save=save_plots)

        logger.info("\n" + report["classification_report"])
        logger.info("✅ Full evaluation complete.")
        return report
