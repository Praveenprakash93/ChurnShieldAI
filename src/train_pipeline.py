"""
train_pipeline.py - Full Model Training Pipeline
=================================================
Author: Data Science Team
Project: Customer Churn Prediction
Description: End-to-end training runner. Executes all pipeline steps:
             data loading → preprocessing → model training →
             evaluation → explainability → save artifacts.

Run from project root:
    python src/train_pipeline.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path Setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import logger, CONFIG
from data_loader import load_and_clean_data
from preprocessing import preprocess_data
from model_training import train_and_select_best
from evaluation import ModelEvaluator
from explainability import ChurnExplainer


def run_full_pipeline(tune_model: bool = False) -> dict:
    """
    Execute the complete end-to-end churn prediction pipeline.

    Args:
        tune_model: Whether to run hyperparameter tuning on best model.

    Returns:
        Dictionary with pipeline results summary.
    """
    logger.info("=" * 70)
    logger.info("🚀 CHURN PREDICTION PIPELINE — STARTING")
    logger.info("=" * 70)

    # ── STEP 1 & 2: Load and Clean Data ───────────────────────────────────
    logger.info("\n" + "─" * 70)
    df, loader = load_and_clean_data()

    churn_stats = loader.get_churn_statistics()
    logger.info(f"📊 Dataset: {churn_stats['total_customers']:,} customers | "
                f"Churn Rate: {churn_stats['churn_rate']}%")

    # ── STEP 3: Preprocessing ─────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    preprocessor.save()
    logger.info(f"✅ Preprocessor saved. Features: {len(preprocessor.feature_names)}")

    # ── STEP 4: Model Training ────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    best_name, best_model, results_df, trainer = train_and_select_best(
        X_train, X_test, y_train, y_test, tune=tune_model
    )

    # ── STEP 5: Evaluation ────────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    evaluator = ModelEvaluator(best_model, model_name=best_name)
    eval_report = evaluator.full_evaluation_report(
        X_train, X_test, y_train, y_test,
        all_models=trainer.trained_models,
        save_plots=True
    )

    # Model comparison plot
    from evaluation import ModelEvaluator as ME
    ME.plot_model_comparison(results_df)

    # ── STEP 6: Explainability ────────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    try:
        explainer = ChurnExplainer(
            model=best_model,
            feature_names=preprocessor.feature_names,
            model_name=best_name
        )
        explainer.run_explainability_pipeline(X_train, X_test, save=True)
    except Exception as e:
        logger.warning(f"⚠️  SHAP explainability failed: {e}. Skipping.")

    # ── Summary ───────────────────────────────────────────────────────────
    metrics = eval_report["metrics"]
    cv = eval_report["cv_results"]

    summary = {
        "best_model": best_name,
        "accuracy": metrics["Accuracy"],
        "f1_score": metrics["F1 Score"],
        "roc_auc": metrics["ROC-AUC"],
        "cv_f1_mean": cv["mean"],
        "cv_f1_std": cv["std"],
        "total_features": len(preprocessor.feature_names),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    logger.info("\n" + "=" * 70)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Best Model:      {best_name}")
    logger.info(f"  Accuracy:        {metrics['Accuracy']:.2f}%")
    logger.info(f"  F1 Score:        {metrics['F1 Score']:.2f}%")
    logger.info(f"  ROC-AUC:         {metrics['ROC-AUC']:.2f}%")
    logger.info(f"  CV F1:           {cv['mean']:.2f}% ± {cv['std']:.2f}%")
    logger.info("=" * 70)
    logger.info("\n📁 Artifacts saved:")
    logger.info(f"  Model    → {CONFIG['model_path']}")
    logger.info(f"  Scaler   → {CONFIG['scaler_path']}")
    logger.info(f"  Features → {CONFIG['feature_names_path']}")
    logger.info(f"  Plots    → reports/figures/")
    logger.info("\n🚀 Run the dashboard:")
    logger.info("  streamlit run app/app.py")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Training Pipeline"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run hyperparameter tuning (slower but better performance)"
    )
    args = parser.parse_args()

    summary = run_full_pipeline(tune_model=args.tune)
