# 🛡️ ChurnShield AI — Customer Churn Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-brightgreen?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A production-ready, end-to-end Customer Churn Prediction system with AI-powered insights, real-time risk scoring, and business recommendations.**

[🚀 Live Demo](#installation) • [📊 Results](#results) • [🔬 SHAP Explainability](#model-explainability) • [📖 Documentation](#project-structure)

</div>

---

## 📋 Table of Contents

- [Business Problem](#business-problem)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Explainability](#model-explainability)
- [Dashboard Features](#dashboard-features)
- [Business Recommendations Engine](#business-recommendations-engine)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🏢 Business Problem

Customer churn — when a customer stops using a company's service — is one of the most critical challenges in the telecom industry.

**The Cost of Churn:**
- Acquiring a new customer costs **5–25x** more than retaining an existing one
- A **5% reduction** in churn can increase profitability by **25–95%**
- The IBM Telco dataset shows a **26.5% churn rate** — translating to **$139,000+/month** in lost revenue

**Our Solution**: Build an intelligent, explainable ML system that:
1. **Predicts** which customers are likely to churn (before they leave)
2. **Explains** why (using SHAP values)
3. **Recommends** targeted retention strategies

---

## 🎯 Project Overview

```
End-to-End ML Pipeline:
  Raw Data → Cleaning → EDA → Feature Engineering → Model Training
  → Evaluation → SHAP Explainability → Streamlit Dashboard → Business Insights
```

**Key Achievements:**
| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| Accuracy | 81.4% |
| F1 Score | 63.9% |
| ROC-AUC | 87.8% |
| Cross-Val F1 | 63.2% ± low std |
| Features Used | 35+ engineered features |
| Models Compared | 7 classifiers |

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- **Source**: [IBM Developer](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Size**: 7,043 customers × 21 features
- **Target**: `Churn` (Yes/No → 1/0)
- **Class Imbalance**: 73.5% No Churn / 26.5% Churn

### Features

| Category | Features |
|----------|----------|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Account | Tenure, Contract, PaperlessBilling, PaymentMethod |
| Services | PhoneService, MultipleLines, InternetService |
| Add-ons | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport |
| Streaming | StreamingTV, StreamingMovies |
| Charges | MonthlyCharges, TotalCharges |
| Engineered | AvgMonthlyRevenue, ServicesCount, IsNewCustomer, ChargesPerService |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ChurnShield AI Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────────────┐ │
│  │  Raw CSV │───▶│ DataLoader │───▶│    Data Cleaning         │ │
│  │  IBM     │    │ (cached)   │    │  - Dedupe, nulls, types  │ │
│  └──────────┘    └────────────┘    └──────────┬───────────────┘ │
│                                               │                 │
│                                    ┌──────────▼───────────────┐ │
│                                    │   Feature Engineering     │ │
│                                    │  - Binary Encoding        │ │
│                                    │  - One-Hot Encoding       │ │
│                                    │  - StandardScaling        │ │
│                                    │  - Derived Features       │ │
│                                    └──────────┬───────────────┘ │
│                                               │                 │
│              ┌────────────────────────────────▼───────────────┐ │
│              │            Model Training (7 Models)           │ │
│              │  LR │ DT │ RF │ GB │ XGBoost │ SVM │ KNN      │ │
│              └────────────────────────────────┬───────────────┘ │
│                                               │                 │
│         ┌────────────────────┬────────────────▼──────────────┐  │
│         │    Evaluation      │       SHAP Explainability      │  │
│         │  Confusion Matrix  │   Feature Importance          │  │
│         │  ROC / PR Curves   │   Waterfall / Force Plot      │  │
│         │  Cross Validation  │   Individual Explanations     │  │
│         └────────────────────┴───────────────────────────────┘  │
│                                               │                 │
│                              ┌────────────────▼───────────────┐ │
│                              │    Streamlit Dashboard          │ │
│                              │  Real-time Predictions         │ │
│                              │  Risk Score + Gauge Chart      │ │
│                              │  Business Recommendations      │ │
│                              │  Revenue Loss Estimator        │ │
│                              └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|----------|-----------|---------|
| Language | Python | 3.12+ |
| ML Framework | scikit-learn | 1.5.0 |
| Boosting | XGBoost | 2.0.3 |
| Explainability | SHAP | 0.45.1 |
| Data | Pandas, NumPy | 2.2.2, 1.26.4 |
| Visualization | Plotly, Matplotlib, Seaborn | Latest |
| Dashboard | Streamlit | 1.35.0 |
| Serialization | Joblib | 1.4.2 |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── 📂 data/
│   ├── raw/                    # Original IBM dataset (auto-downloaded)
│   └── processed/              # Cleaned & feature-engineered data
│
├── 📓 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis (20+ plots)
│   └── 02_Model_Training.ipynb # Model training, comparison & evaluation
│
├── 🐍 src/
│   ├── utils.py                # Config, logging, risk scoring, recommendations
│   ├── data_loader.py          # Data loading, cleaning, validation
│   ├── preprocessing.py        # Encoding, scaling, train/test split
│   ├── feature_engineering.py  # Feature selection, KMeans segmentation
│   ├── model_training.py       # 7 model training, comparison, selection
│   ├── evaluation.py           # Metrics, confusion matrix, ROC, PR curves
│   ├── explainability.py       # SHAP explainer (tree/linear/kernel)
│   ├── predict.py              # Inference pipeline, batch prediction
│   └── train_pipeline.py       # End-to-end training orchestrator
│
├── 📱 app/
│   └── app.py                  # Streamlit dashboard (1200+ lines)
│
├── 🤖 models/
│   ├── churn_model.pkl         # Best trained model (XGBoost)
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── feature_names.json      # Feature metadata
│
├── 📊 reports/
│   └── figures/                # All generated plots (PNG)
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- pip or conda

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction/Customer-Churn-Prediction
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Or using conda
conda create -n churn-env python=3.12
conda activate churn-env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
# Option A: Quick training (no hyperparameter tuning)
python src/train_pipeline.py

# Option B: Full training with hyperparameter tuning (recommended)
python src/train_pipeline.py --tune
```

This will:
- ✅ Auto-download the IBM Telco dataset
- ✅ Clean and preprocess data
- ✅ Train 7 ML models
- ✅ Select and save best model (XGBoost)
- ✅ Generate all evaluation plots
- ✅ Run SHAP explainability

### Step 5: Launch Dashboard

```bash
streamlit run app/app.py
```

Open your browser: **http://localhost:8501**

---

## 📖 Usage

### Dashboard Workflow

1. **Fill in customer details** in the sidebar form
2. **Click "Predict Churn Risk"** — instant AI prediction
3. **View results**:
   - Churn probability (%)
   - Risk score (0–100)
   - Animated gauge chart
   - Revenue loss estimate
   - Business recommendations
4. **Explore tabs** for dataset insights, model comparison, SHAP

### Batch Prediction (API)

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()
predictor.load_artifacts()

# Single customer
result = predictor.predict_single({
    "gender": "Male",
    "SeniorCitizen": "No",
    "tenure": 2,
    "Contract": "Month-to-month",
    "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
    # ... other fields
})

print(f"Churn: {result['churn_prediction']}")
print(f"Risk Score: {result['risk_score']}/100")
print(f"Probability: {result['churn_probability_pct']:.1f}%")

# Batch prediction
import pandas as pd
df = pd.read_csv("data/raw/telco_customer_churn.csv")
predictions_df = predictor.predict_batch(df)
```

### Custom Training

```python
from src.data_loader import load_and_clean_data
from src.preprocessing import preprocess_data
from src.model_training import train_and_select_best

# Load & clean
df, loader = load_and_clean_data()

# Preprocess
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Train all 7 models & select best
best_name, best_model, results_df, trainer = train_and_select_best(
    X_train, X_test, y_train, y_test, tune=True
)
```

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV F1 |
|-------|----------|-----------|--------|----------|---------|-------|
| Logistic Regression | 80.7% | 66.1% | 54.3% | 59.7% | 84.2% | 59.1% |
| Decision Tree | 78.2% | 59.4% | 60.3% | 59.8% | 73.8% | 58.4% |
| Random Forest | 82.1% | 70.3% | 56.2% | 62.4% | 87.1% | 61.8% |
| Gradient Boosting | 80.9% | 67.4% | 58.1% | 62.4% | 86.3% | 61.5% |
| **XGBoost ⭐** | **81.4%** | 68.8% | **59.7%** | **63.9%** | **87.8%** | **63.2%** |
| SVM | 79.8% | 65.1% | 55.8% | 60.1% | 85.9% | 59.8% |
| KNN | 77.6% | 60.8% | 53.4% | 56.9% | 79.1% | 55.7% |

### Why XGBoost?

> For imbalanced churn prediction, **F1 Score** is more important than Accuracy.
> F1 balances Precision (avoiding false alarms) and Recall (catching all churners).
> XGBoost wins on F1 (63.9%), ROC-AUC (87.8%), and Cross-Validation consistency.

### Key Business Insights

| Finding | Impact |
|---------|--------|
| Month-to-month contracts → 42.7% churn rate | 15x higher than 2-year contracts |
| Fiber optic internet → 41.9% churn | Investigate pricing perception |
| Low tenure (< 12 mo) → High risk | Onboarding programs critical |
| High monthly charges → Higher churn | Price sensitivity lever |
| No OnlineSecurity → 30% higher churn | Bundle security add-ons |

---

## 🔬 Model Explainability

SHAP (SHapley Additive exPlanations) provides transparent AI predictions.

### Top Churn Drivers (Global)

| Rank | Feature | SHAP Impact | Business Meaning |
|------|---------|-------------|-----------------|
| 1 | Contract_Month-to-month | +0.312 | No commitment → easy to leave |
| 2 | tenure | -0.289 | Longer customer → less likely to churn |
| 3 | TotalCharges | -0.198 | Higher lifetime value → invested |
| 4 | MonthlyCharges | +0.187 | Price sensitivity |
| 5 | InternetService_Fiber | +0.156 | Dissatisfied premium users |
| 6 | PaymentMethod_E-check | +0.143 | Convenience/engagement signal |
| 7 | OnlineSecurity_No | +0.127 | Less sticky without add-ons |

---

## 🖥️ Dashboard Features

### Prediction Panel
- 🔮 **Real-time churn prediction** with probability score
- 📊 **Animated gauge chart** (0–100 risk score)
- 🎯 **Probability breakdown** bar chart
- 💰 **Revenue loss estimator** (monthly, annual, worst-case)

### Analytics Panels
- 📈 **Prediction history** with trend chart & CSV download
- 📊 **Dataset insights** with 4 interactive Plotly charts
- 🤖 **Model comparison** table + radar chart
- 🔬 **SHAP explainability** — feature importance + interpretation

### Business Intelligence
- 💡 **7 smart recommendations** based on customer profile
- 🚨 **Priority-based action items** (Critical/High/Medium/Low)
- 👥 **Customer segmentation** context

---

## 💼 Business Recommendations Engine

The system generates personalized retention strategies:

| Customer Profile | Recommendation |
|-----------------|----------------|
| Month-to-month contract | Offer 15-20% discount for annual upgrade |
| Monthly charges > $70 | Personalized bundle discount |
| Tenure < 12 months | Onboarding loyalty reward package |
| No Online Security | Free 3-month security trial |
| Churn prob > 80% | Immediate escalation to retention team |
| Senior Citizen | Dedicated senior support package |

---

## 🔮 Future Improvements

- [ ] **Real-time Data Pipeline** — Kafka/Spark streaming integration
- [ ] **A/B Testing Framework** — Test retention strategy effectiveness
- [ ] **LLM Integration** — GPT-4 powered personalized email generation
- [ ] **REST API** — FastAPI deployment for microservices integration
- [ ] **Docker Containerization** — Easy cloud deployment (AWS/GCP/Azure)
- [ ] **MLflow Tracking** — Experiment tracking and model registry
- [ ] **Deep Learning** — LSTM for sequential customer behavior
- [ ] **Automated Retraining** — Scheduled model refresh with new data
- [ ] **Ensemble Stacking** — Meta-learner combining all 7 models
- [ ] **Customer Lifetime Value** — Prioritize retention by CLV

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contact & Portfolio

Built as a **Data Science Portfolio Project** for campus placements at TCS, Infosys, Accenture, Deloitte, Capgemini, and similar firms.

**Skills Demonstrated:**
- End-to-end ML pipeline design
- Feature engineering & selection
- Multi-model comparison & selection
- Model explainability (SHAP)
- Production-quality Python code
- Streamlit dashboard development
- Business-oriented data science thinking

---

<div align="center">
⭐ Star this repo if it helped you! | Built with ❤️ and Python 🐍
</div>
