"""
explainability.py - Model Explainability with SHAP
====================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Uses SHAP (SHapley Additive exPlanations) to explain
             model predictions globally and individually.
             Generates feature importance, waterfall, force plots.
"""

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import CONFIG, FIGURES_DIR, logger, timer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT THEME
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0E1117"
CARD_BG = "#1A1F2E"
ACCENT = "#00D4FF"
DANGER = "#FF6B6B"
SUCCESS = "#4ECDC4"
TEXT = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ChurnExplainer:
    """
    SHAP-based model explainability for churn prediction.

    Supports:
        - TreeExplainer for tree-based models (XGBoost, RF, GBM)
        - LinearExplainer for linear models (Logistic Regression)
        - KernelExplainer as fallback for any model (SVM, KNN)

    Attributes:
        model: Fitted sklearn estimator.
        explainer: SHAP explainer object.
        shap_values: Computed SHAP values for the dataset.
        feature_names (List[str]): Names of features.
    """

    # Model types that support TreeExplainer
    TREE_MODELS = [
        "XGBClassifier", "RandomForestClassifier",
        "GradientBoostingClassifier", "DecisionTreeClassifier",
        "ExtraTreesClassifier"
    ]
    # Model types that support LinearExplainer
    LINEAR_MODELS = ["LogisticRegression", "LinearSVC"]

    def __init__(self,
                 model: Any,
                 feature_names: List[str],
                 model_name: str = "Model") -> None:
        """
        Initialize ChurnExplainer.

        Args:
            model: Fitted sklearn estimator.
            feature_names: List of feature names.
            model_name: Display name of the model.
        """
        self.model = model
        self.feature_names = feature_names
        self.model_name = model_name
        self.explainer = None
        self.shap_values: Optional[np.ndarray] = None
        self._background_data: Optional[pd.DataFrame] = None
        logger.info(f"ChurnExplainer initialized for: {model_name}")

    # ─────────────────────────────────────────────
    # BUILD EXPLAINER
    # ─────────────────────────────────────────────
    @timer
    def build_explainer(self,
                        X_train: pd.DataFrame,
                        n_background: int = 100) -> None:
        """
        Select and build the appropriate SHAP explainer.

        Args:
            X_train: Training data (used as background for KernelExplainer).
            n_background: Number of background samples for KernelExplainer.
        """
        model_type = type(self.model).__name__
        self._background_data = X_train

        logger.info(f"Building SHAP explainer for {model_type}...")

        try:
            if model_type in self.TREE_MODELS:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("✅ Using TreeExplainer.")

            elif model_type in self.LINEAR_MODELS:
                background = shap.maskers.Independent(X_train, max_samples=n_background)
                self.explainer = shap.LinearExplainer(self.model, background)
                logger.info("✅ Using LinearExplainer.")

            else:
                # KernelExplainer (model-agnostic but slower)
                background = shap.kmeans(X_train, min(50, n_background))
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict,
                    background
                )
                logger.info("✅ Using KernelExplainer (model-agnostic).")

        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}). Falling back to KernelExplainer.")
            background = shap.kmeans(X_train, min(50, n_background))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )

    # ─────────────────────────────────────────────
    # COMPUTE SHAP VALUES
    # ─────────────────────────────────────────────
    @timer
    def compute_shap_values(self,
                            X: pd.DataFrame,
                            max_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for a dataset.

        Args:
            X: Feature matrix to explain.
            max_samples: Maximum rows to compute SHAP values for
                         (reduces computation time for large datasets).

        Returns:
            Array of SHAP values shape (n_samples, n_features).
        """
        if self.explainer is None:
            raise RuntimeError("Build explainer first with build_explainer().")

        # Limit to max_samples for speed
        X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")

        raw = self.explainer.shap_values(X_sample)

        # Handle multi-output SHAP values
        # RF TreeExplainer may return:
        #   - list of 2 arrays: [class0_shap, class1_shap] → take index [1]
        #   - 3D ndarray: (n_samples, n_features, n_classes) → take [:, :, 1]
        #   - 2D ndarray: (n_samples, n_features) → use directly
        if isinstance(raw, list) and len(raw) == 2:
            self.shap_values = raw[1]  # Churn class (1)
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            self.shap_values = raw[:, :, 1]  # Churn class (last dim)
        else:
            self.shap_values = raw

        self._X_sample = X_sample
        logger.info(f"✅ SHAP values computed. Shape: {self.shap_values.shape}")
        return self.shap_values

    # ─────────────────────────────────────────────
    # FEATURE IMPORTANCE PLOT (Global)
    # ─────────────────────────────────────────────
    @timer
    def plot_feature_importance(self,
                                top_n: int = 20,
                                save: bool = True,
                                filename: str = "shap_feature_importance.png"
                                ) -> plt.Figure:
        """
        Plot global feature importance using mean absolute SHAP values.

        Args:
            top_n: Number of top features to display.
            save: Whether to save figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        if self.shap_values is None:
            raise RuntimeError("Compute SHAP values first.")

        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "Feature": self.feature_names[:len(mean_abs_shap)],
            "SHAP Importance": mean_abs_shap
        }).sort_values("SHAP Importance", ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(DARK_BG)

        # Color gradient
        n = len(feature_importance)
        colors = plt.cm.cool(np.linspace(0.2, 1.0, n))

        bars = ax.barh(
            feature_importance["Feature"],
            feature_importance["SHAP Importance"],
            color=colors, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5
        )

        # Value labels
        for bar, val in zip(bars, feature_importance["SHAP Importance"]):
            ax.text(bar.get_width() + 0.0005,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=8, color=TEXT)

        ax.set_xlabel("Mean |SHAP Value| (Impact on Churn Prediction)",
                      fontsize=12, labelpad=10)
        ax.set_title(
            f"Global Feature Importance — {self.model_name}\n"
            f"(Top {top_n} Features by SHAP Value)",
            fontsize=14, fontweight="bold", pad=15, color=TEXT
        )
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ Feature importance plot saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # SHAP SUMMARY PLOT (Beeswarm)
    # ─────────────────────────────────────────────
    @timer
    def plot_shap_summary(self,
                          top_n: int = 20,
                          save: bool = True,
                          filename: str = "shap_summary.png") -> plt.Figure:
        """
        Generate SHAP summary (beeswarm) plot.

        Args:
            top_n: Number of features to show.
            save: Whether to save figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        if self.shap_values is None:
            raise RuntimeError("Compute SHAP values first.")

        fig = plt.figure(figsize=(12, 9))
        fig.patch.set_facecolor(DARK_BG)

        # SHAP summary plot
        feature_names_short = [
            f[:30] for f in self.feature_names[:self.shap_values.shape[1]]
        ]
        shap.summary_plot(
            self.shap_values,
            self._X_sample,
            feature_names=feature_names_short,
            max_display=top_n,
            show=False,
            plot_type="dot",
            color_bar=True,
        )

        plt.title(f"SHAP Summary Plot — {self.model_name}",
                  fontsize=14, fontweight="bold", pad=15, color=TEXT)
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ SHAP summary plot saved → {path}")
        return fig

    # ─────────────────────────────────────────────
    # WATERFALL PLOT (Individual Prediction)
    # ─────────────────────────────────────────────
    @timer
    def plot_waterfall(self,
                       customer_index: int = 0,
                       save: bool = True,
                       filename: str = "shap_waterfall.png") -> plt.Figure:
        """
        Plot SHAP waterfall for a single customer prediction.

        Args:
            customer_index: Row index in _X_sample to explain.
            save: Whether to save figure.
            filename: Output filename.

        Returns:
            matplotlib Figure.
        """
        if self.shap_values is None:
            raise RuntimeError("Compute SHAP values first.")

        if customer_index >= len(self._X_sample):
            customer_index = 0

        try:
            # Modern SHAP API
            explanation = shap.Explanation(
                values=self.shap_values[customer_index],
                base_values=self.explainer.expected_value
                if not isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value[1],
                data=self._X_sample.iloc[customer_index].values,
                feature_names=self.feature_names[:self.shap_values.shape[1]]
            )

            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor(DARK_BG)
            shap.waterfall_plot(explanation, max_display=15, show=False)
            plt.title(
                f"SHAP Waterfall — Customer #{customer_index + 1}",
                fontsize=14, fontweight="bold", color=TEXT
            )
            plt.tight_layout()

        except Exception as e:
            logger.warning(f"Waterfall plot failed with new API: {e}. "
                           "Falling back to bar plot.")
            fig = self._plot_individual_bar(customer_index)

        if save:
            path = FIGURES_DIR / filename
            fig.savefig(path, dpi=CONFIG["figure_dpi"],
                        bbox_inches="tight", facecolor=DARK_BG)
            logger.info(f"✅ Waterfall plot saved → {path}")
        return fig

    def _plot_individual_bar(self, customer_index: int) -> plt.Figure:
        """Fallback: bar chart of SHAP values for a single customer."""
        values = self.shap_values[customer_index]
        features = self.feature_names[:len(values)]

        df = pd.DataFrame({"Feature": features, "SHAP": values})
        df = df.reindex(df["SHAP"].abs().sort_values(ascending=False).index)
        df = df.head(15).sort_values("SHAP")

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(DARK_BG)
        colors = [DANGER if v > 0 else SUCCESS for v in df["SHAP"]]
        ax.barh(df["Feature"], df["SHAP"], color=colors, alpha=0.85)
        ax.axvline(x=0, color=TEXT, lw=1.5, linestyle="--")
        ax.set_title(
            f"Individual SHAP Values — Customer #{customer_index + 1}",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("SHAP Value (impact on churn probability)")
        plt.tight_layout()
        return fig

    # ─────────────────────────────────────────────
    # GET TOP FEATURES FOR A CUSTOMER
    # ─────────────────────────────────────────────
    def explain_customer(self,
                         customer_features: pd.DataFrame,
                         top_n: int = 5) -> pd.DataFrame:
        """
        Explain prediction for a single customer (for Streamlit dashboard).

        Args:
            customer_features: Single-row DataFrame with processed features.
            top_n: Number of top contributing features to return.

        Returns:
            DataFrame with feature name, value, and SHAP contribution.
        """
        if self.explainer is None:
            raise RuntimeError("Build explainer first.")

        raw = self.explainer.shap_values(customer_features)

        if isinstance(raw, list) and len(raw) == 2:
            shap_vals = raw[1][0]
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            shap_vals = raw[0, :, 1]  # first sample, all features, churn class
        elif len(raw.shape) == 2:
            shap_vals = raw[0]
        else:
            shap_vals = raw

        features = self.feature_names[:len(shap_vals)]
        df = pd.DataFrame({
            "Feature": features,
            "Value": customer_features.iloc[0, :len(features)].values,
            "SHAP Contribution": shap_vals
        })
        df["Abs SHAP"] = df["SHAP Contribution"].abs()
        df = df.sort_values("Abs SHAP", ascending=False).head(top_n)
        df["Direction"] = df["SHAP Contribution"].apply(
            lambda x: "↑ Increases Churn Risk" if x > 0
            else "↓ Decreases Churn Risk"
        )
        return df[["Feature", "Value", "SHAP Contribution", "Direction"]].reset_index(drop=True)

    # ─────────────────────────────────────────────
    # FULL EXPLAINABILITY PIPELINE
    # ─────────────────────────────────────────────
    @timer
    def run_explainability_pipeline(self,
                                    X_train: pd.DataFrame,
                                    X_test: pd.DataFrame,
                                    save: bool = True) -> None:
        """
        Run the full SHAP explainability pipeline.

        Args:
            X_train: Training data (background for explainer).
            X_test: Test data (compute SHAP values for).
            save: Whether to save all plots.
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Model Explainability (SHAP)")
        logger.info("=" * 60)

        self.build_explainer(X_train)
        self.compute_shap_values(X_test)
        self.plot_feature_importance(save=save)
        self.plot_shap_summary(save=save)
        self.plot_waterfall(customer_index=0, save=save)

        logger.info("✅ Explainability pipeline complete.")
