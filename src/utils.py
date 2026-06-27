"""
utils.py - Utility Functions and Configuration
===============================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Shared utilities, logging setup, configuration constants,
             and helper functions used across the project.
"""

import os
import json
import logging
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# SUPPRESS WARNINGS (Production Mode)
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
                  REPORTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    # Dataset
    "dataset_url": (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    ),
    "dataset_filename": "telco_customer_churn.csv",
    "target_column": "Churn",
    "customer_id_column": "customerID",

    # Features
    "categorical_features": [
        "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ],
    "binary_features": [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling"
    ],
    "numerical_features": [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ],

    # ML Configuration
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "n_clusters": 4,

    # Model paths
    "model_path": str(MODELS_DIR / "churn_model.pkl"),
    "scaler_path": str(MODELS_DIR / "scaler.pkl"),
    "encoder_path": str(MODELS_DIR / "encoder.pkl"),
    "feature_names_path": str(MODELS_DIR / "feature_names.json"),

    # Thresholds
    "churn_threshold": 0.5,
    "high_risk_threshold": 0.80,
    "medium_risk_threshold": 0.50,

    # Business Rules
    "business_rules": {
        "high_monthly_charges": 70.0,
        "low_tenure_months": 12,
        "discount_offer_pct": 20,
    },

    # Visualization
    "plot_style": "dark_background",
    "color_palette": ["#00D4FF", "#FF6B6B", "#4ECDC4", "#45B7D1",
                      "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"],
    "churn_colors": {"No": "#4ECDC4", "Yes": "#FF6B6B"},
    "figure_dpi": 150,
    "figure_size": (12, 8),
}

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
def setup_logger(name: str = "churn_prediction",
                 level: int = logging.INFO,
                 log_to_file: bool = True) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        log_to_file: Whether to write logs to file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# Global logger instance
logger = setup_logger()

# ─────────────────────────────────────────────
# TIMER DECORATOR
# ─────────────────────────────────────────────
def timer(func):
    """
    Decorator that logs the execution time of a function.

    Args:
        func: Function to time.

    Returns:
        Wrapped function with timing.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"⏱  {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper

# ─────────────────────────────────────────────
# DATA VALIDATION HELPERS
# ─────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame,
                       required_cols: List[str]) -> Tuple[bool, str]:
    """
    Validate a DataFrame has required columns and is not empty.

    Args:
        df: DataFrame to validate.
        required_cols: List of required column names.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if df is None or df.empty:
        return False, "DataFrame is None or empty."

    missing = set(required_cols) - set(df.columns)
    if missing:
        return False, f"Missing required columns: {missing}"

    return True, "OK"


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of missing values per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: ['Column', 'Missing Count', 'Missing %'].
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": pct.values
    })
    return summary[summary["Missing Count"] > 0].sort_values(
        "Missing %", ascending=False
    ).reset_index(drop=True)


def get_data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of column data types and unique value counts.

    Args:
        df: Input DataFrame.

    Returns:
        Summary DataFrame.
    """
    return pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
        "Sample Values": [df[c].dropna().unique()[:3].tolist()
                          for c in df.columns]
    })

# ─────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────
def calculate_risk_score(churn_probability: float) -> int:
    """
    Convert churn probability to a 0–100 risk score.

    Args:
        churn_probability: Float between 0 and 1.

    Returns:
        Integer risk score between 0 and 100.
    """
    return min(100, max(0, int(round(churn_probability * 100))))


def get_risk_category(risk_score: int) -> Tuple[str, str]:
    """
    Categorize a risk score into a label and color.

    Args:
        risk_score: Integer between 0 and 100.

    Returns:
        Tuple of (category_label, hex_color).
    """
    if risk_score >= 80:
        return "🔴 High Risk", "#FF4444"
    elif risk_score >= 50:
        return "🟠 Medium Risk", "#FF8C00"
    elif risk_score >= 25:
        return "🟡 Low Risk", "#FFD700"
    else:
        return "🟢 Very Low Risk", "#00C851"

# ─────────────────────────────────────────────
# BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────
def generate_business_recommendations(
    customer_data: Dict[str, Any],
    churn_probability: float
) -> List[Dict[str, str]]:
    """
    Generate actionable business recommendations based on customer profile.

    Args:
        customer_data: Dictionary with customer feature values.
        churn_probability: Predicted churn probability.

    Returns:
        List of recommendation dictionaries with 'title', 'detail', 'priority'.
    """
    recommendations = []
    rules = CONFIG["business_rules"]

    # Rule 1: Month-to-Month Contract
    contract = customer_data.get("Contract", "")
    if contract == "Month-to-month":
        recommendations.append({
            "title": "📋 Upgrade to Annual Contract",
            "detail": (
                "This customer is on a Month-to-month contract, which has "
                "the highest churn rate (42%). Offer a 15-20% discount to "
                "switch to a 1-year or 2-year plan. Long-term contracts "
                "reduce churn risk by 3x."
            ),
            "priority": "High",
            "icon": "📋"
        })

    # Rule 2: High Monthly Charges
    monthly_charges = float(customer_data.get("MonthlyCharges", 0))
    if monthly_charges > rules["high_monthly_charges"]:
        recommendations.append({
            "title": "💰 Offer Discounted Bundle Plan",
            "detail": (
                f"Monthly charges of ${monthly_charges:.0f} are above the "
                f"high-value threshold (${rules['high_monthly_charges']:.0f}). "
                f"Offer a {rules['discount_offer_pct']}% loyalty discount or "
                "bundle promotion to retain this customer."
            ),
            "priority": "High",
            "icon": "💰"
        })

    # Rule 3: Low Tenure (New Customer)
    tenure = int(customer_data.get("tenure", 0))
    if tenure <= rules["low_tenure_months"]:
        recommendations.append({
            "title": "🎁 Onboarding Loyalty Reward",
            "detail": (
                f"Customer has only {tenure} months of tenure. Early-stage "
                "customers are at high risk. Send a personalized onboarding "
                "kit, offer a 30-day free premium add-on, and assign a "
                "dedicated customer success manager."
            ),
            "priority": "Medium",
            "icon": "🎁"
        })

    # Rule 4: No Online Security
    if customer_data.get("OnlineSecurity", "No") == "No":
        recommendations.append({
            "title": "🔒 Free Security Upgrade Trial",
            "detail": (
                "Customers without OnlineSecurity churn 30% more. Offer a "
                "3-month free trial of the Online Security add-on. This adds "
                "perceived value and increases stickiness."
            ),
            "priority": "Medium",
            "icon": "🔒"
        })

    # Rule 5: No Tech Support
    if customer_data.get("TechSupport", "No") == "No":
        recommendations.append({
            "title": "🛠️ Complimentary Tech Support",
            "detail": (
                "Customers without TechSupport have higher dissatisfaction "
                "scores. Offer 2 free tech support sessions or a 30-day "
                "trial to demonstrate value."
            ),
            "priority": "Low",
            "icon": "🛠️"
        })

    # Rule 6: High Churn Probability - Escalate
    if churn_probability >= CONFIG["high_risk_threshold"]:
        recommendations.append({
            "title": "🚨 Immediate Retention Escalation",
            "detail": (
                f"Churn probability is critically high at "
                f"{churn_probability*100:.1f}%. Escalate to the retention "
                "team immediately. Assign a senior account manager, offer "
                "a personalized retention deal, and schedule a call within "
                "24 hours. Every day of delay increases churn risk."
            ),
            "priority": "Critical",
            "icon": "🚨"
        })

    # Rule 7: Senior Citizen
    # SeniorCitizen may arrive as "Yes"/"No" (from sidebar) or 0/1 (after encoding)
    _senior_raw = customer_data.get("SeniorCitizen", 0)
    _is_senior = (
        str(_senior_raw).strip().lower() in ("yes", "1", "true")
        if isinstance(_senior_raw, str)
        else int(_senior_raw) == 1
    )
    if _is_senior:
        recommendations.append({
            "title": "👴 Senior-Friendly Support Package",
            "detail": (
                "Senior citizens (65+) often struggle with digital services. "
                "Offer a dedicated senior helpline, simplified billing, and "
                "a personal account advisor to improve satisfaction."
            ),
            "priority": "Medium",
            "icon": "👴"
        })

    # Sort by priority
    priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

    return recommendations

# ─────────────────────────────────────────────
# REVENUE LOSS PREDICTION
# ─────────────────────────────────────────────
def estimate_revenue_loss(
    monthly_charges: float,
    churn_probability: float,
    tenure_remaining_estimate: int = 12
) -> Dict[str, float]:
    """
    Estimate potential revenue loss if customer churns.

    Args:
        monthly_charges: Customer's monthly charges.
        churn_probability: Predicted churn probability.
        tenure_remaining_estimate: Estimated months remaining.

    Returns:
        Dictionary with revenue loss estimates.
    """
    expected_loss = monthly_charges * tenure_remaining_estimate * churn_probability
    worst_case = monthly_charges * tenure_remaining_estimate
    cac_savings = 300.0  # Average customer acquisition cost

    return {
        "expected_monthly_loss": round(monthly_charges * churn_probability, 2),
        "expected_annual_loss": round(expected_loss, 2),
        "worst_case_loss": round(worst_case, 2),
        "retention_roi": round(cac_savings + expected_loss, 2),
        "break_even_discount": round(
            (expected_loss / (expected_loss + cac_savings)) * 100, 1
        )
    }

# ─────────────────────────────────────────────
# FILE I/O HELPERS
# ─────────────────────────────────────────────
def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """Save data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON → {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_number(value: float, prefix: str = "",
                  suffix: str = "", decimals: int = 2) -> str:
    """Format a number with prefix/suffix for display."""
    return f"{prefix}{value:,.{decimals}f}{suffix}"


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTATION LABELS
# ─────────────────────────────────────────────
SEGMENT_LABELS = {
    0: {"name": "Champions", "color": "#00C851",
        "description": "Low churn risk, high value, long tenure"},
    1: {"name": "At Risk", "color": "#FF4444",
        "description": "High churn risk, needs immediate attention"},
    2: {"name": "Promising", "color": "#FFD700",
        "description": "New customers with growth potential"},
    3: {"name": "Need Attention", "color": "#FF8C00",
        "description": "Medium risk, engage proactively"},
}
