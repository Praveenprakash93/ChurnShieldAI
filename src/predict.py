"""
predict.py - Prediction Inference Pipeline
===========================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Production-ready prediction pipeline that loads trained
             models and preprocessors, and generates churn predictions
             with risk scores for new customer data.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from utils import (
    CONFIG, MODELS_DIR, logger,
    calculate_risk_score, get_risk_category,
    generate_business_recommendations, estimate_revenue_loss
)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ChurnPredictor:
    """
    End-to-end churn prediction inference engine.

    Loads trained model, scaler, and feature metadata, then
    provides predict() for single or batch predictions.

    Attributes:
        model: Loaded sklearn estimator.
        scaler: Loaded StandardScaler.
        feature_names (List[str]): Feature names expected by the model.
        ohe_categories (Dict): OHE categories for categorical columns.
        is_ready (bool): Whether all artifacts are loaded.
    """

    # Binary encoding maps (same as preprocessing.py)
    BINARY_MAP = {
        "gender": {"Male": 1, "Female": 0},
        "Partner": {"Yes": 1, "No": 0},
        "Dependents": {"Yes": 1, "No": 0},
        "PhoneService": {"Yes": 1, "No": 0},
        "PaperlessBilling": {"Yes": 1, "No": 0},
        "SeniorCitizen": {"Yes": 1, "No": 0},
    }
    OHE_COLS = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
    ]
    NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    def __init__(self) -> None:
        """Initialize predictor (artifacts loaded lazily)."""
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.ohe_categories: Dict = {}
        self.is_ready: bool = False
        logger.info("ChurnPredictor initialized.")

    # ─────────────────────────────────────────────
    # LOAD ARTIFACTS
    # ─────────────────────────────────────────────
    def load_artifacts(self,
                       model_path: Optional[str] = None,
                       scaler_path: Optional[str] = None,
                       feature_names_path: Optional[str] = None) -> bool:
        """
        Load all trained artifacts from disk.

        Args:
            model_path: Path to churn_model.pkl.
            scaler_path: Path to scaler.pkl.
            feature_names_path: Path to feature_names.json.

        Returns:
            True if all artifacts loaded successfully.
        """
        model_path = model_path or CONFIG["model_path"]
        scaler_path = scaler_path or CONFIG["scaler_path"]
        feature_names_path = (feature_names_path
                               or CONFIG["feature_names_path"])

        try:
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded: {type(self.model).__name__}")

            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded.")

            with open(feature_names_path, "r") as f:
                meta = json.load(f)
            self.feature_names = meta["feature_names"]
            self.ohe_categories = meta.get("ohe_categories", {})
            logger.info(f"✅ Feature metadata loaded. "
                        f"Features: {len(self.feature_names)}")

            self.is_ready = True
            return True

        except FileNotFoundError as e:
            logger.error(f"❌ Artifact not found: {e}")
            logger.error("Run the training pipeline first to generate models.")
            self.is_ready = False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")
            self.is_ready = False
            return False

    # ─────────────────────────────────────────────
    # TRANSFORM SINGLE CUSTOMER INPUT
    # ─────────────────────────────────────────────
    def _transform_input(self, raw_input: Dict) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to a raw customer dictionary.

        This mirrors the ChurnPreprocessor.transform() logic but
        operates on a single record dictionary.

        Args:
            raw_input: Dictionary with customer feature values.

        Returns:
            Processed single-row DataFrame ready for prediction.
        """
        df = pd.DataFrame([raw_input])

        # ── Feature Engineering ───────────────────────────────────────────
        tenure = float(df.get("tenure", [0])[0])
        monthly = float(df.get("MonthlyCharges", [0])[0])
        total = float(df.get("TotalCharges", [monthly])[0])

        df["AvgMonthlyRevenue"] = round(total / (tenure + 1), 2)
        df["IsNewCustomer"] = int(tenure <= 12)

        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        service_count = sum(
            1 for col in service_cols
            if df.get(col, ["No"])[0] == "Yes"
        )
        df["ServicesCount"] = service_count
        df["ChargesPerService"] = round(monthly / (service_count + 1), 2)

        # ── Binary Encoding ───────────────────────────────────────────────
        for col, mapping in self.BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        # ── One-Hot Encoding ──────────────────────────────────────────────
        for col in self.OHE_COLS:
            if col not in df.columns:
                continue
            categories = self.ohe_categories.get(col, [])
            for cat in categories:
                df[f"{col}_{cat}"] = int(df[col].values[0] == cat)
            df = df.drop(columns=[col])

        # ── Scaling ───────────────────────────────────────────────────────
        num_cols = [c for c in self.NUMERICAL_COLS + [
            "AvgMonthlyRevenue", "ChargesPerService"
        ] if c in df.columns]
        df[num_cols] = self.scaler.transform(df[num_cols])

        # ── Align to training feature set ──────────────────────────────────
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]

        return df

    # ─────────────────────────────────────────────
    # SINGLE CUSTOMER PREDICTION
    # ─────────────────────────────────────────────
    def predict_single(
        self,
        customer_data: Dict[str, Any],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict churn probability for a single customer.

        Args:
            customer_data: Dictionary with customer features.
            threshold: Decision threshold for binary prediction.

        Returns:
            Dictionary with prediction results and recommendations.

        Raises:
            RuntimeError: If artifacts are not loaded.
        """
        if not self.is_ready:
            raise RuntimeError("Predictor not ready. Call load_artifacts() first.")

        # Preprocess
        X = self._transform_input(customer_data)

        # Predict
        churn_prob = float(self.model.predict_proba(X)[0, 1])
        churn_pred = int(churn_prob >= threshold)
        risk_score = calculate_risk_score(churn_prob)
        risk_label, risk_color = get_risk_category(risk_score)

        # Business recommendations
        recommendations = generate_business_recommendations(
            customer_data, churn_prob
        )

        # Revenue loss estimate
        monthly_charges = float(customer_data.get("MonthlyCharges", 0))
        revenue = estimate_revenue_loss(monthly_charges, churn_prob)

        result = {
            "churn_prediction": bool(churn_pred),
            "churn_probability": round(churn_prob, 4),
            "churn_probability_pct": round(churn_prob * 100, 1),
            "not_churn_probability": round(1 - churn_prob, 4),
            "risk_score": risk_score,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "recommendations": recommendations,
            "revenue_impact": revenue,
            "threshold_used": threshold,
        }

        logger.info(
            f"Prediction: {'⚠️ CHURN' if churn_pred else '✅ RETAIN'} | "
            f"Probability: {churn_prob*100:.1f}% | "
            f"Risk Score: {risk_score}/100"
        )
        return result

    # ─────────────────────────────────────────────
    # BATCH PREDICTION
    # ─────────────────────────────────────────────
    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Predict churn for a batch of customers.

        Args:
            df: DataFrame with customer features (raw, unprocessed).
            threshold: Decision threshold.

        Returns:
            Input DataFrame with added prediction columns.
        """
        if not self.is_ready:
            raise RuntimeError("Predictor not ready. Call load_artifacts() first.")

        logger.info(f"Batch prediction for {len(df):,} customers...")

        results = []
        for _, row in df.iterrows():
            try:
                pred = self.predict_single(row.to_dict(), threshold)
                results.append({
                    "churn_probability": pred["churn_probability_pct"],
                    "churn_prediction": pred["churn_prediction"],
                    "risk_score": pred["risk_score"],
                    "risk_label": pred["risk_label"],
                })
            except Exception as e:
                results.append({
                    "churn_probability": None,
                    "churn_prediction": None,
                    "risk_score": None,
                    "risk_label": "Error",
                })
                logger.error(f"Prediction error for row: {e}")

        result_df = df.copy()
        pred_df = pd.DataFrame(results)
        for col in pred_df.columns:
            result_df[col] = pred_df[col].values

        logger.info("✅ Batch prediction complete.")
        return result_df


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PREDICTOR INSTANCE (Singleton for Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
_predictor_instance: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """
    Return a singleton ChurnPredictor instance with artifacts loaded.

    Returns:
        Ready ChurnPredictor instance.
    """
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_ready:
        _predictor_instance = ChurnPredictor()
        _predictor_instance.load_artifacts()
    return _predictor_instance


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Demo prediction
    sample_customer = {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 179.00,
    }

    predictor = get_predictor()
    if predictor.is_ready:
        result = predictor.predict_single(sample_customer)
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Churn Prediction: {'YES' if result['churn_prediction'] else 'NO'}")
        print(f"Churn Probability: {result['churn_probability_pct']:.1f}%")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Risk Level: {result['risk_label']}")
        print(f"\nRevenue Impact: {result['revenue_impact']}")
        print(f"\nRecommendations ({len(result['recommendations'])}):")
        for rec in result["recommendations"]:
            print(f"  [{rec['priority']}] {rec['title']}")
    else:
        print("❌ Models not found. Please run the training pipeline first.")
