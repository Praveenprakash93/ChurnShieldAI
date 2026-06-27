"""
feature_engineering.py - Feature Engineering and Selection
===========================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Advanced feature engineering, feature selection using
             statistical tests and tree-based importance, and
             customer segmentation using KMeans clustering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, f_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from utils import CONFIG, MODELS_DIR, logger, timer, SEGMENT_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION CLASS
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSelector:
    """
    Multi-method feature selection for the churn prediction pipeline.

    Methods supported:
        - Correlation-based filtering
        - Mutual Information
        - ANOVA F-statistic
        - Random Forest feature importance
        - Recursive Feature Elimination (RFE)

    Attributes:
        selected_features (List[str]): Final selected feature names.
        importance_df (pd.DataFrame): Feature importance scores.
    """

    def __init__(self, k_best: int = 20) -> None:
        """
        Initialize FeatureSelector.

        Args:
            k_best: Number of top features to select via SelectKBest.
        """
        self.k_best = k_best
        self.selected_features: List[str] = []
        self.importance_df: Optional[pd.DataFrame] = None
        self._rf_selector: Optional[RandomForestClassifier] = None
        logger.info(f"FeatureSelector initialized. k_best={k_best}")

    def correlation_filter(self,
                           X: pd.DataFrame,
                           threshold: float = 0.90) -> List[str]:
        """
        Remove highly correlated features (Pearson correlation > threshold).

        Args:
            X: Feature matrix.
            threshold: Correlation threshold above which one of a pair is dropped.

        Returns:
            List of feature names after removing correlated features.
        """
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_tri.columns
                   if any(upper_tri[col] > threshold)]

        if to_drop:
            logger.info(f"Correlation filter: removing {len(to_drop)} "
                        f"highly correlated features: {to_drop[:5]}...")

        return [c for c in X.columns if c not in to_drop]

    def mutual_info_selection(self,
                              X: pd.DataFrame,
                              y: pd.Series) -> pd.DataFrame:
        """
        Compute mutual information between each feature and target.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            DataFrame with features and mutual information scores.
        """
        mi_scores = mutual_info_classif(
            X, y, random_state=CONFIG["random_state"]
        )
        return pd.DataFrame({
            "Feature": X.columns,
            "MI_Score": mi_scores
        }).sort_values("MI_Score", ascending=False).reset_index(drop=True)

    def rf_importance_selection(self,
                                X: pd.DataFrame,
                                y: pd.Series) -> pd.DataFrame:
        """
        Use Random Forest feature importance for selection.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            DataFrame with features and RF importance scores.
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        rf.fit(X, y)
        self._rf_selector = rf

        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "RF_Importance": rf.feature_importances_
        }).sort_values("RF_Importance", ascending=False).reset_index(drop=True)

        logger.info(f"Top 10 features by RF importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['Feature']:<35} {row['RF_Importance']:.4f}")

        return importance_df

    @timer
    def fit_select(self,
                   X: pd.DataFrame,
                   y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run all selection methods and return the combined importance table
        and the final selected feature matrix.

        Args:
            X: Feature matrix.
            y: Target series.

        Returns:
            Tuple of (X_selected, importance_summary_df).
        """
        logger.info("=" * 60)
        logger.info("Feature Selection")
        logger.info("=" * 60)

        # Step 1: Correlation filter
        uncorrelated = self.correlation_filter(X)
        X_filtered = X[uncorrelated]

        # Step 2: RF importance
        rf_df = self.rf_importance_selection(X_filtered, y)

        # Step 3: Mutual information
        mi_df = self.mutual_info_selection(X_filtered, y)

        # Step 4: Combine scores
        combined = rf_df.merge(mi_df, on="Feature", how="outer").fillna(0)
        combined["Combined_Score"] = (
            0.6 * combined["RF_Importance"] +
            0.4 * combined["MI_Score"] / (combined["MI_Score"].max() + 1e-8)
        )
        combined = combined.sort_values(
            "Combined_Score", ascending=False
        ).reset_index(drop=True)

        # Step 5: Select top k features
        top_k = min(self.k_best, len(combined))
        self.selected_features = combined.head(top_k)["Feature"].tolist()
        self.importance_df = combined

        X_selected = X_filtered[self.selected_features]
        logger.info(f"✅ Selected {len(self.selected_features)} features.")
        return X_selected, combined


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOMER SEGMENTATION CLASS
# ─────────────────────────────────────────────────────────────────────────────
class CustomerSegmentation:
    """
    KMeans-based customer segmentation for churn analysis.

    Segments customers into groups (Champions, At Risk, Promising,
    Need Attention) based on behavioral features.

    Attributes:
        kmeans (KMeans): Fitted KMeans model.
        n_clusters (int): Number of customer segments.
        segment_labels (Dict): Mapping of cluster id to segment info.
    """

    # Features used for segmentation (business-relevant)
    SEGMENTATION_FEATURES = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "AvgMonthlyRevenue", "ServicesCount", "IsNewCustomer",
    ]

    def __init__(self, n_clusters: int = 4) -> None:
        """
        Initialize CustomerSegmentation.

        Args:
            n_clusters: Number of KMeans clusters.
        """
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.scaler = MinMaxScaler()
        self.segment_labels = SEGMENT_LABELS
        self._feature_cols: List[str] = []
        logger.info(f"CustomerSegmentation initialized. n_clusters={n_clusters}")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and scale segmentation features."""
        available = [f for f in self.SEGMENTATION_FEATURES
                     if f in df.columns]
        self._feature_cols = available
        X = df[available].copy().fillna(0)
        return X

    @timer
    def fit(self, df: pd.DataFrame) -> "CustomerSegmentation":
        """
        Fit KMeans on segmentation features.

        Args:
            df: Clean DataFrame (pre-OHE, with engineered features).

        Returns:
            Self (fitted).
        """
        logger.info("Fitting CustomerSegmentation KMeans...")
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=CONFIG["random_state"],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(X_scaled)
        logger.info(f"✅ KMeans fitted. "
                    f"Inertia: {self.kmeans.inertia_:.2f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict segment labels for new data.

        Args:
            df: Input DataFrame with required features.

        Returns:
            Array of cluster labels.
        """
        if self.kmeans is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def assign_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add segment label column to DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with 'Segment', 'SegmentName' columns added.
        """
        df = df.copy()
        labels = self.predict(df)
        df["Segment"] = labels
        df["SegmentName"] = [
            self.segment_labels.get(l, {}).get("name", f"Cluster {l}")
            for l in labels
        ]
        return df

    def get_segment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return churn rate and average stats per segment.

        Args:
            df: DataFrame with 'Segment' and 'Churn' columns.

        Returns:
            Summary DataFrame per segment.
        """
        if "Segment" not in df.columns:
            df = self.assign_segments(df)

        agg_cols = {
            "Churn": ["count", "sum", "mean"],
            "MonthlyCharges": "mean",
            "tenure": "mean",
        }
        if "TotalCharges" in df.columns:
            agg_cols["TotalCharges"] = "mean"

        summary = df.groupby("SegmentName").agg(agg_cols).round(2)
        summary.columns = [
            "Total Customers", "Churned", "Churn Rate",
            "Avg Monthly Charges", "Avg Tenure"
        ][:len(summary.columns)]
        summary["Churn Rate"] = (summary["Churn Rate"] * 100).round(1)
        summary = summary.reset_index()
        return summary

    def save(self, path: Optional[str] = None) -> None:
        """Save fitted KMeans to disk."""
        path = path or str(MODELS_DIR / "kmeans.pkl")
        joblib.dump({"kmeans": self.kmeans, "scaler": self.scaler,
                     "features": self._feature_cols}, path)
        logger.info(f"✅ Segmentation model saved → {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CustomerSegmentation":
        """Load fitted segmentation model from disk."""
        path = path or str(MODELS_DIR / "kmeans.pkl")
        data = joblib.load(path)
        seg = cls()
        seg.kmeans = data["kmeans"]
        seg.scaler = data["scaler"]
        seg._feature_cols = data["features"]
        logger.info(f"✅ Segmentation model loaded from {path}")
        return seg
