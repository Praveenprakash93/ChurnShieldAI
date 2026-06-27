"""
data_loader.py - Data Loading and Cleaning Module
==================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: Handles loading the IBM Telco Customer Churn dataset from URL
             or local cache, performs comprehensive data cleaning, and
             returns a clean, analysis-ready DataFrame.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    CONFIG, logger, timer, validate_dataframe,
    check_missing_values, get_data_types_summary,
    DATA_RAW_DIR
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RAW_FILE_PATH = DATA_RAW_DIR / CONFIG["dataset_filename"]

# Column renames for consistency (lowercase)
COLUMN_RENAME_MAP = {
    "customerID": "customerID",
    "gender": "gender",
    "SeniorCitizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "tenure",
    "PhoneService": "PhoneService",
    "MultipleLines": "MultipleLines",
    "InternetService": "InternetService",
    "OnlineSecurity": "OnlineSecurity",
    "OnlineBackup": "OnlineBackup",
    "DeviceProtection": "DeviceProtection",
    "TechSupport": "TechSupport",
    "StreamingTV": "StreamingTV",
    "StreamingMovies": "StreamingMovies",
    "Contract": "Contract",
    "PaperlessBilling": "PaperlessBilling",
    "PaymentMethod": "PaymentMethod",
    "MonthlyCharges": "MonthlyCharges",
    "TotalCharges": "TotalCharges",
    "Churn": "Churn",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATA LOADER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """
    Handles loading, caching, and cleaning the IBM Telco Customer Churn dataset.

    Attributes:
        url (str): Remote URL of the dataset.
        local_path (Path): Local cache path for the raw CSV.
        df_raw (pd.DataFrame): Raw loaded DataFrame.
        df_clean (pd.DataFrame): Cleaned DataFrame.
    """

    def __init__(self,
                 url: Optional[str] = None,
                 local_path: Optional[Path] = None) -> None:
        """
        Initialize DataLoader with dataset URL and local cache path.

        Args:
            url: Dataset download URL. Defaults to CONFIG url.
            local_path: Local path to cache/load CSV. Defaults to data/raw/.
        """
        self.url = url or CONFIG["dataset_url"]
        self.local_path = local_path or RAW_FILE_PATH
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None

        logger.info("DataLoader initialized.")
        logger.info(f"Local cache path: {self.local_path}")

    # ─────────────────────────────────────────────
    # STEP 1: LOAD DATA
    # ─────────────────────────────────────────────
    @timer
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from local cache or download from URL.

        Returns:
            Raw DataFrame.

        Raises:
            RuntimeError: If both local load and download fail.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Dataset")
        logger.info("=" * 60)

        # Try loading from local cache first
        if self.local_path.exists():
            logger.info(f"Loading from local cache: {self.local_path}")
            try:
                self.df_raw = pd.read_csv(self.local_path)
                logger.info(f"✅ Loaded {len(self.df_raw):,} rows × "
                            f"{len(self.df_raw.columns)} columns from cache.")
                return self.df_raw
            except Exception as e:
                logger.warning(f"Local load failed: {e}. Attempting download.")

        # Download from URL
        logger.info(f"Downloading dataset from: {self.url}")
        try:
            urllib.request.urlretrieve(self.url, self.local_path)
            self.df_raw = pd.read_csv(self.local_path)
            logger.info(f"✅ Downloaded and loaded {len(self.df_raw):,} rows × "
                        f"{len(self.df_raw.columns)} columns.")
            return self.df_raw
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    # ─────────────────────────────────────────────
    # STEP 2: CLEAN DATA
    # ─────────────────────────────────────────────
    @timer
    def clean_data(self) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning on the raw dataset.

        Cleaning steps:
            1. Validate required columns exist
            2. Remove duplicates
            3. Fix TotalCharges (spaces → NaN → float)
            4. Handle missing values
            5. Convert SeniorCitizen 0/1 → "No"/"Yes"
            6. Strip whitespace from string columns
            7. Encode target variable (Yes/No → 1/0)
            8. Reset index

        Returns:
            Cleaned DataFrame.

        Raises:
            ValueError: If raw data has not been loaded first.
        """
        if self.df_raw is None:
            raise ValueError("Raw data not loaded. Call load_data() first.")

        logger.info("=" * 60)
        logger.info("STEP 2: Data Cleaning")
        logger.info("=" * 60)

        df = self.df_raw.copy()
        initial_rows = len(df)

        # ── 2.1 Validate columns ────────────────────────────────────────────
        required_cols = list(COLUMN_RENAME_MAP.keys())
        is_valid, msg = validate_dataframe(df, required_cols)
        if not is_valid:
            raise ValueError(f"Data validation failed: {msg}")
        logger.info(f"✅ All {len(required_cols)} required columns present.")

        # ── 2.2 Remove duplicates ───────────────────────────────────────────
        dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        logger.info(f"✅ Removed {dupes} duplicate rows. "
                    f"Remaining: {len(df):,}")

        # ── 2.3 Fix TotalCharges (contains spaces) ──────────────────────────
        # IBM dataset has empty strings ' ' in TotalCharges for new customers
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        tc_nulls = df["TotalCharges"].isnull().sum()
        logger.info(f"✅ Converted TotalCharges to numeric. "
                    f"NaN introduced: {tc_nulls}")

        # ── 2.4 Handle Missing Values ───────────────────────────────────────
        # TotalCharges NaN → impute with tenure × MonthlyCharges
        mask = df["TotalCharges"].isnull()
        df.loc[mask, "TotalCharges"] = (
            df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
        )
        logger.info(f"✅ Imputed {mask.sum()} TotalCharges NaN values.")

        # Check for remaining nulls
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            logger.warning(f"⚠️  {remaining_nulls} null values remain.")
            # Drop rows with any remaining NaN
            df = df.dropna()
            logger.info(f"Dropped rows with NaN. Remaining: {len(df):,}")

        # ── 2.5 Strip whitespace from object columns ────────────────────────
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        logger.info(f"✅ Stripped whitespace from {len(str_cols)} string cols.")

        # ── 2.6 SeniorCitizen: 0/1 → "No"/"Yes" ────────────────────────────
        df["SeniorCitizen"] = df["SeniorCitizen"].map(
            {0: "No", 1: "Yes"}
        ).fillna(df["SeniorCitizen"])
        logger.info("✅ SeniorCitizen encoded as No/Yes.")

        # ── 2.7 Standardize Churn column ────────────────────────────────────
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        logger.info("✅ Target column 'Churn' encoded: Yes=1, No=0.")

        # ── 2.8 Validate data types ──────────────────────────────────────────
        df["tenure"] = df["tenure"].astype(int)
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        # ── 2.9 Reset index ──────────────────────────────────────────────────
        df = df.reset_index(drop=True)

        # Summary
        dropped = initial_rows - len(df)
        logger.info(f"✅ Cleaning complete. "
                    f"Rows: {initial_rows:,} → {len(df):,} "
                    f"(dropped {dropped}).")

        self.df_clean = df
        return df

    # ─────────────────────────────────────────────
    # STEP 3: DESCRIPTIVE STATISTICS
    # ─────────────────────────────────────────────
    def get_descriptive_stats(self) -> pd.DataFrame:
        """
        Generate comprehensive descriptive statistics.

        Returns:
            DataFrame with descriptive statistics.
        """
        if self.df_clean is None:
            raise ValueError("Clean data not available. Call clean_data() first.")

        logger.info("Generating descriptive statistics...")
        return self.df_clean.describe(include="all").T

    def get_churn_statistics(self) -> dict:
        """
        Return churn-specific summary statistics.

        Returns:
            Dictionary with churn counts, rates, and revenue impact.
        """
        if self.df_clean is None:
            raise ValueError("Clean data not available.")

        df = self.df_clean
        total = len(df)
        churned = df["Churn"].sum()
        not_churned = total - churned
        churn_rate = churned / total * 100

        avg_monthly_churn = df[df["Churn"] == 1]["MonthlyCharges"].mean()
        avg_monthly_retain = df[df["Churn"] == 0]["MonthlyCharges"].mean()
        monthly_revenue_at_risk = (
            df[df["Churn"] == 1]["MonthlyCharges"].sum()
        )

        return {
            "total_customers": total,
            "churned": int(churned),
            "retained": int(not_churned),
            "churn_rate": round(churn_rate, 2),
            "retention_rate": round(100 - churn_rate, 2),
            "avg_monthly_charges_churned": round(avg_monthly_churn, 2),
            "avg_monthly_charges_retained": round(avg_monthly_retain, 2),
            "monthly_revenue_at_risk": round(monthly_revenue_at_risk, 2),
        }

    def get_missing_value_report(self) -> pd.DataFrame:
        """Return missing value summary for raw data."""
        if self.df_raw is None:
            raise ValueError("Raw data not loaded.")
        return check_missing_values(self.df_raw)

    def get_data_type_report(self) -> pd.DataFrame:
        """Return data type summary for raw data."""
        if self.df_raw is None:
            raise ValueError("Raw data not loaded.")
        return get_data_types_summary(self.df_raw)

    def detect_outliers(self) -> pd.DataFrame:
        """
        Detect outliers in numerical columns using IQR method.

        Returns:
            DataFrame summarizing outlier counts per numerical column.
        """
        if self.df_clean is None:
            raise ValueError("Clean data not available.")

        num_cols = CONFIG["numerical_features"]
        results = []
        for col in num_cols:
            if col not in self.df_clean.columns:
                continue
            series = self.df_clean[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((series < lower) | (series > upper)).sum()
            results.append({
                "Column": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outlier Count": outliers,
                "Outlier %": round(outliers / len(self.df_clean) * 100, 2),
            })

        return pd.DataFrame(results)

    # ─────────────────────────────────────────────
    # FULL PIPELINE
    # ─────────────────────────────────────────────
    @timer
    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the full data loading and cleaning pipeline.

        Returns:
            Cleaned DataFrame ready for EDA and modeling.
        """
        logger.info("🚀 Starting Data Loading Pipeline...")
        self.load_data()
        self.df_clean = self.clean_data()
        stats = self.get_churn_statistics()
        logger.info(f"📊 Churn Rate: {stats['churn_rate']}% "
                    f"({stats['churned']:,} churned / "
                    f"{stats['total_customers']:,} total)")
        logger.info("✅ Data Pipeline Complete.")
        return self.df_clean


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean_data(
    url: Optional[str] = None,
    local_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, DataLoader]:
    """
    Convenience function: load and clean data, return df and loader.

    Args:
        url: Optional custom URL.
        local_path: Optional local path.

    Returns:
        Tuple of (cleaned_dataframe, loader_instance).
    """
    loader = DataLoader(url=url, local_path=local_path)
    df = loader.run_pipeline()
    return df, loader


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, loader = load_and_clean_data()
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nMissing Values:\n{loader.get_missing_value_report()}")
    print(f"\nOutliers:\n{loader.detect_outliers()}")
    print(f"\nChurn Stats:\n{loader.get_churn_statistics()}")
