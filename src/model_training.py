"""
model_training.py - Machine Learning Model Training
=====================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Trains multiple ML classifiers, compares them with a
             comprehensive metrics table, selects and saves the best model.
             Includes hyperparameter tuning for the best model.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import CONFIG, MODELS_DIR, logger, timer

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_model_definitions() -> Dict[str, Any]:
    """
    Return a dictionary of all models with their configurations.

    Returns:
        Dict mapping model name → sklearn estimator instance.
    """
    rs = CONFIG["random_state"]
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=rs,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=rs,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=rs,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=rs
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=rs,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            random_state=rs,
            class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="minkowski",
            n_jobs=-1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ModelTrainer:
    """
    Trains, evaluates, and compares multiple ML classifiers.

    Attributes:
        models (Dict): Model definitions.
        results (List[Dict]): Metrics for each trained model.
        best_model_name (str): Name of the selected best model.
        best_model: Fitted best model estimator.
        trained_models (Dict): All fitted model estimators.
    """

    def __init__(self) -> None:
        """Initialize trainer with model definitions."""
        self.models = get_model_definitions()
        self.results: List[Dict] = []
        self.best_model_name: str = ""
        self.best_model = None
        self.trained_models: Dict[str, Any] = {}
        self._cv = StratifiedKFold(
            n_splits=CONFIG["cv_folds"],
            shuffle=True,
            random_state=CONFIG["random_state"]
        )
        logger.info(f"ModelTrainer initialized with {len(self.models)} models.")

    # ─────────────────────────────────────────────
    # EVALUATE ONE MODEL
    # ─────────────────────────────────────────────
    def _evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """
        Train and evaluate a single model, returning all metrics.

        Args:
            model_name: Name of the model.
            model: Unfitted sklearn estimator.
            X_train, X_test: Train and test feature matrices.
            y_train, y_test: Train and test target series.

        Returns:
            Dictionary with all evaluation metrics.
        """
        logger.info(f"  Training: {model_name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = (
            roc_auc_score(y_test, y_prob)
            if y_prob is not None else np.nan
        )

        # Cross-validation F1
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self._cv,
            scoring="f1",
            n_jobs=-1
        )

        result = {
            "Model": model_name,
            "Accuracy": round(accuracy * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1 * 100, 2),
            "ROC-AUC": round(roc_auc * 100, 2) if not np.isnan(roc_auc) else "N/A",
            "CV F1 Mean": round(cv_scores.mean() * 100, 2),
            "CV F1 Std": round(cv_scores.std() * 100, 2),
        }

        logger.info(
            f"    ✅ Acc={accuracy*100:.1f}% | F1={f1*100:.1f}% | "
            f"AUC={roc_auc*100:.1f}% | CV F1={cv_scores.mean()*100:.1f}%"
        )
        return result

    # ─────────────────────────────────────────────
    # TRAIN ALL MODELS
    # ─────────────────────────────────────────────
    @timer
    def train_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Train all models and collect comparison metrics.

        Args:
            X_train, X_test: Feature matrices.
            y_train, y_test: Target series.

        Returns:
            DataFrame with model comparison results.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Training All Models")
        logger.info("=" * 60)

        self.results = []
        models_fresh = get_model_definitions()  # Fresh instances

        for name, model in models_fresh.items():
            try:
                result = self._evaluate_model(
                    name, model, X_train, X_test, y_train, y_test
                )
                self.results.append(result)
                self.trained_models[name] = model
            except Exception as e:
                logger.error(f"  ❌ {name} failed: {e}")
                continue

        results_df = pd.DataFrame(self.results)
        logger.info("\n" + results_df.to_string(index=False))
        return results_df

    # ─────────────────────────────────────────────
    # SELECT BEST MODEL
    # ─────────────────────────────────────────────
    def select_best_model(
        self,
        metric: str = "F1 Score",
        results_df: Optional[pd.DataFrame] = None
    ) -> Tuple[str, Any]:
        """
        Select the best model based on a given metric.

        Selection rationale for Churn prediction:
            - F1 Score is preferred over Accuracy because the dataset is
              imbalanced (~26% churn). F1 balances Precision & Recall,
              ensuring we catch actual churners (high Recall) while not
              flooding support with false alarms (high Precision).
            - ROC-AUC is used as a secondary validation metric.

        Args:
            metric: Column name from results_df to rank by.
            results_df: Optional pre-computed results DataFrame.

        Returns:
            Tuple of (best_model_name, best_model_estimator).
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if results_df.empty:
            raise ValueError("No results available. Run train_all() first.")

        # Filter out non-numeric metric rows
        df_valid = results_df[results_df[metric] != "N/A"].copy()
        df_valid[metric] = pd.to_numeric(df_valid[metric])
        best_row = df_valid.loc[df_valid[metric].idxmax()]

        self.best_model_name = best_row["Model"]
        self.best_model = self.trained_models[self.best_model_name]

        logger.info("=" * 60)
        logger.info(f"🏆 Best Model: {self.best_model_name}")
        logger.info(f"   {metric}: {best_row[metric]:.2f}%")
        logger.info("=" * 60)
        logger.info("WHY THIS MODEL?")
        logger.info("  - F1 Score optimized for imbalanced churn data.")
        logger.info("  - Balances Precision (cost of false positives) and")
        logger.info("    Recall (cost of missing actual churners).")
        logger.info("  - Cross-validation confirms generalization ability.")
        logger.info("  - XGBoost/GBM ensembles handle feature interactions well.")

        return self.best_model_name, self.best_model

    # ─────────────────────────────────────────────
    # HYPERPARAMETER TUNING (Best Model)
    # ─────────────────────────────────────────────
    @timer
    def tune_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Run GridSearchCV on the best model to fine-tune hyperparameters.

        Args:
            X_train: Training feature matrix.
            y_train: Training target.

        Returns:
            Best fitted estimator after tuning.
        """
        logger.info(f"Tuning {self.best_model_name}...")

        param_grids = {
            "XGBoost": {
                "n_estimators": [150, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.8],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [6, 8, 10],
                "min_samples_leaf": [3, 5, 10],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [4, 5, 6],
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        }

        param_grid = param_grids.get(self.best_model_name)
        if param_grid is None:
            logger.info(f"No param grid for {self.best_model_name}. Skipping tuning.")
            return self.best_model

        grid_search = GridSearchCV(
            estimator=self.best_model,
            param_grid=param_grid,
            cv=self._cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.trained_models[self.best_model_name] = self.best_model

        logger.info(f"✅ Best params: {grid_search.best_params_}")
        logger.info(f"   Best CV F1: {grid_search.best_score_*100:.2f}%")

        return self.best_model

    # ─────────────────────────────────────────────
    # SAVE BEST MODEL
    # ─────────────────────────────────────────────
    def save_best_model(self, path: Optional[str] = None) -> str:
        """
        Save the best fitted model to disk.

        Args:
            path: Optional save path. Defaults to CONFIG model_path.

        Returns:
            Path where model was saved.
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model().")

        path = path or CONFIG["model_path"]
        joblib.dump(self.best_model, path)
        logger.info(f"✅ Best model ({self.best_model_name}) saved → {path}")
        return path

    @staticmethod
    def load_model(path: Optional[str] = None):
        """Load a saved model from disk."""
        path = path or CONFIG["model_path"]
        model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_select_best(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tune: bool = False
) -> Tuple[str, Any, pd.DataFrame, "ModelTrainer"]:
    """
    Full training pipeline: train all models, select best, optionally tune.

    Args:
        X_train, X_test: Feature splits.
        y_train, y_test: Target splits.
        tune: Whether to run GridSearchCV on best model.

    Returns:
        Tuple of (best_name, best_model, results_df, trainer).
    """
    trainer = ModelTrainer()
    results_df = trainer.train_all(X_train, X_test, y_train, y_test)
    best_name, best_model = trainer.select_best_model(results_df=results_df)

    if tune:
        trainer.tune_best_model(X_train, y_train)

    trainer.save_best_model()
    return best_name, best_model, results_df, trainer


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_clean_data
    from preprocessing import preprocess_data

    df, _ = load_and_clean_data()
    X_train, X_test, y_train, y_test, prep = preprocess_data(df)

    best_name, best_model, results_df, trainer = train_and_select_best(
        X_train, X_test, y_train, y_test, tune=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\n🏆 Best Model: {best_name}")
