"""
preprocessing.py - Data Preprocessing and Feature Engineering Pipeline
========================================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Encodes categorical features, scales numerical features,
             splits data into train/test sets, and saves preprocessors.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import CONFIG, MODELS_DIR, DATA_PROCESSED_DIR, logger, timer


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ChurnPreprocessor:
    """
    Comprehensive preprocessing pipeline for the Churn dataset.

    Handles:
        - Binary encoding (Yes/No → 1/0)
        - One-Hot Encoding for multi-class categoricals
        - Standard scaling for numerical features
        - Train/test split
        - Saving/loading of fitted preprocessors

    Attributes:
        scaler (StandardScaler): Fitted scaler for numerical features.
        feature_names (List[str]): Final feature names after encoding.
        categorical_cols (List[str]): Columns to one-hot encode.
        binary_cols (List[str]): Columns to binary encode.
        numerical_cols (List[str]): Columns to scale.
    """

    # Columns that have Yes/No or Male/Female semantics
    BINARY_MAP = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "SeniorCitizen": {"Yes": 1, "No": 0},
    }

    # Multi-class categoricals → OHE
    OHE_COLS = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
    ]

    # Numerical features to scale
    NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    def __init__(self) -> None:
        """Initialize the preprocessor with default state."""
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.ohe_categories: Dict[str, List[str]] = {}
        self._is_fitted: bool = False
        logger.info("ChurnPreprocessor initialized.")

    # ─────────────────────────────────────────────
    # BINARY ENCODING
    # ─────────────────────────────────────────────
    def _apply_binary_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply binary encoding to Yes/No and gender columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with binary-encoded columns.
        """
        df = df.copy()
        for col, mapping in self.BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
                logger.debug(f"Binary encoded: {col}")
        return df

    # ─────────────────────────────────────────────
    # ONE-HOT ENCODING
    # ─────────────────────────────────────────────
    def _apply_ohe(self, df: pd.DataFrame,
                   fit: bool = True) -> pd.DataFrame:
        """
        Apply One-Hot Encoding to multi-class categorical columns.

        Args:
            df: Input DataFrame.
            fit: If True, learn categories from data (training).
                 If False, use stored categories (inference).

        Returns:
            DataFrame with OHE columns replacing original categorical cols.
        """
        df = df.copy()
        ohe_cols = [c for c in self.OHE_COLS if c in df.columns]

        if fit:
            # Learn categories
            self.ohe_categories = {
                col: sorted(df[col].unique().tolist())
                for col in ohe_cols
            }

        # Apply get_dummies with known categories
        for col in ohe_cols:
            categories = self.ohe_categories.get(col, [])
            for cat in categories:
                new_col = f"{col}_{cat}"
                df[new_col] = (df[col] == cat).astype(int)
            df = df.drop(columns=[col])

        return df

    # ─────────────────────────────────────────────
    # FEATURE ENGINEERING (Additional Features)
    # ─────────────────────────────────────────────
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional derived features for better predictive power.

        New features:
            - AvgMonthlyRevenue: TotalCharges / (tenure + 1)
            - ServicesCount: Total number of add-on services subscribed
            - IsNewCustomer: tenure <= 12 months
            - HasOnlineServices: OnlineSecurity + OnlineBackup bundled
            - ChargesPerService: MonthlyCharges / (ServicesCount + 1)

        Args:
            df: Input DataFrame (post binary encoding).

        Returns:
            DataFrame with additional engineered features.
        """
        df = df.copy()

        # Average monthly revenue
        df["AvgMonthlyRevenue"] = (
            df["TotalCharges"] / (df["tenure"] + 1)
        ).round(2)

        # Is new customer
        df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)

        # Services subscribed count (using raw column values before OHE)
        service_cols_raw = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        service_count = 0
        for col in service_cols_raw:
            if col in df.columns:
                service_count += (df[col] == "Yes").astype(int)
            else:
                # Column may have been OHE'd already; skip
                pass

        if isinstance(service_count, pd.Series):
            df["ServicesCount"] = service_count
        else:
            df["ServicesCount"] = 0

        # Charges per service
        df["ChargesPerService"] = (
            df["MonthlyCharges"] / (df.get("ServicesCount", 1) + 1)
        ).round(2)

        logger.debug("Feature engineering applied.")
        return df

    # ─────────────────────────────────────────────
    # SCALING
    # ─────────────────────────────────────────────
    def _scale_numerical(self, df: pd.DataFrame,
                         fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input DataFrame.
            fit: If True, fit the scaler on data. If False, transform only.

        Returns:
            DataFrame with scaled numerical columns.
        """
        df = df.copy()
        num_cols = [c for c in self.NUMERICAL_COLS
                    if c in df.columns]
        # Include engineered numerical features
        extra_num = ["AvgMonthlyRevenue", "ChargesPerService"]
        for c in extra_num:
            if c in df.columns:
                num_cols.append(c)

        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])

        logger.debug(f"Scaled {len(num_cols)} numerical columns.")
        return df

    # ─────────────────────────────────────────────
    # FULL FIT-TRANSFORM
    # ─────────────────────────────────────────────
    @timer
    def fit_transform(self,
                      df: pd.DataFrame,
                      target_col: str = "Churn") -> Tuple[pd.DataFrame,
                                                          pd.Series]:
        """
        Fit preprocessors on training data and transform it.

        Args:
            df: Input DataFrame with target column.
            target_col: Name of target column.

        Returns:
            Tuple of (feature_matrix, target_series).
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Preprocessing (fit_transform)")
        logger.info("=" * 60)

        df = df.copy()

        # Separate target
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])

        # Drop customer ID
        id_col = CONFIG["customer_id_column"]
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

        # Pipeline steps
        df = self._engineer_features(df)
        df = self._apply_binary_encoding(df)
        df = self._apply_ohe(df, fit=True)
        df = self._scale_numerical(df, fit=True)

        # Store feature names
        self.feature_names = df.columns.tolist()
        self._is_fitted = True

        logger.info(f"✅ fit_transform complete. Features: {len(self.feature_names)}")
        return df, y

    # ─────────────────────────────────────────────
    # TRANSFORM (Inference)
    # ─────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors.

        Args:
            df: Input DataFrame (without target column).

        Returns:
            Transformed feature DataFrame.

        Raises:
            RuntimeError: If preprocessor has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()

        # Drop ID and target if present
        for col in [CONFIG["customer_id_column"], CONFIG["target_column"]]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Pipeline
        df = self._engineer_features(df)
        df = self._apply_binary_encoding(df)
        df = self._apply_ohe(df, fit=False)
        df = self._scale_numerical(df, fit=False)

        # Align columns to training feature set
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]
        return df

    # ─────────────────────────────────────────────
    # TRAIN/TEST SPLIT
    # ─────────────────────────────────────────────
    @staticmethod
    def split_data(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            X: Feature matrix.
            y: Target series.
            test_size: Proportion of test data.
            random_state: Random seed for reproducibility.
            stratify: Whether to stratify by target (maintains class balance).

        Returns:
            X_train, X_test, y_train, y_test
        """
        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )
        logger.info(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")
        logger.info(f"   Train churn rate: "
                    f"{y_train.mean()*100:.1f}%")
        logger.info(f"   Test churn rate:  "
                    f"{y_test.mean()*100:.1f}%")
        return X_train, X_test, y_train, y_test

    # ─────────────────────────────────────────────
    # SAVE / LOAD PREPROCESSOR
    # ─────────────────────────────────────────────
    def save(self,
             scaler_path: Optional[str] = None,
             feature_names_path: Optional[str] = None) -> None:
        """
        Save the fitted scaler and feature names to disk.

        Args:
            scaler_path: Path to save scaler.pkl.
            feature_names_path: Path to save feature_names.json.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")

        scaler_path = scaler_path or CONFIG["scaler_path"]
        feature_names_path = feature_names_path or CONFIG["feature_names_path"]

        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler saved → {scaler_path}")

        meta = {
            "feature_names": self.feature_names,
            "ohe_categories": self.ohe_categories,
            "binary_map": {k: list(v.keys())
                           for k, v in self.BINARY_MAP.items()},
        }
        with open(feature_names_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"✅ Feature names saved → {feature_names_path}")

    @classmethod
    def load(cls,
             scaler_path: Optional[str] = None,
             feature_names_path: Optional[str] = None) -> "ChurnPreprocessor":
        """
        Load a previously fitted preprocessor from disk.

        Args:
            scaler_path: Path to scaler.pkl.
            feature_names_path: Path to feature_names.json.

        Returns:
            Loaded ChurnPreprocessor instance.
        """
        scaler_path = scaler_path or CONFIG["scaler_path"]
        feature_names_path = feature_names_path or CONFIG["feature_names_path"]

        preprocessor = cls()
        preprocessor.scaler = joblib.load(scaler_path)

        with open(feature_names_path, "r") as f:
            meta = json.load(f)

        preprocessor.feature_names = meta["feature_names"]
        preprocessor.ohe_categories = meta["ohe_categories"]
        preprocessor._is_fitted = True

        logger.info(f"✅ Preprocessor loaded from {scaler_path}")
        return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           "ChurnPreprocessor"]:
    """
    Run the full preprocessing pipeline and return train/test splits.

    Args:
        df: Cleaned DataFrame from DataLoader.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )
    return X_train, X_test, y_train, y_test, preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_clean_data

    df, loader = load_and_clean_data()
    X_train, X_test, y_train, y_test, prep = preprocess_data(df)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Features: {len(prep.feature_names)}")
    print(f"Feature names: {prep.feature_names[:10]}...")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
