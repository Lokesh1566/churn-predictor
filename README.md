<div align="center">

# 📊 Customer Churn Prediction System

**End-to-end ML pipeline with REST API & Interactive Dashboard**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

*A production-ready machine learning system that predicts customer churn for telecom companies, featuring automated feature engineering, model comparison, a REST API for real-time predictions, and an interactive Streamlit dashboard.*

[Quick Start](#-quick-start) · [API Docs](#-api-endpoints) · [Dashboard](#-dashboard) · [Architecture](#%EF%B8%8F-architecture) · [Results](#-results)

</div>

---

## 🎯 Problem Statement

Customer churn costs telecom companies **$65.6 billion annually**. Identifying at-risk customers before they leave enables proactive retention strategies. This system predicts which customers are likely to churn using ML classification, providing risk scores and actionable insights.

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates data + trains + saves artifacts)
python src/pipeline.py

# 4. Launch the API
uvicorn api.app:app --reload --port 8000
# → API docs at http://localhost:8000/docs

# 5. Launch the Dashboard
streamlit run streamlit_app/dashboard.py
# → Dashboard at http://localhost:8501
```

**Or with Docker:**
```bash
docker-compose up --build
# API → http://localhost:8000  |  Dashboard → http://localhost:8501
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    DATA LAYER                            │
│  Synthetic Generator  ←→  CSV Storage  ←→  Real Data    │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  ML PIPELINE (src/pipeline.py)           │
│                                                          │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  EDA &   │→│   Feature    │→│  Model Training   │   │
│  │ Cleaning │  │ Engineering  │  │  & Evaluation     │   │
│  └─────────┘  └──────────────┘  └──────────────────┘   │
│                                          │               │
│                    ┌─────────────────────┘               │
│                    ▼                                      │
│           ┌────────────────┐                             │
│           │  Model Artifacts│  (joblib + metadata)       │
│           └────────────────┘                             │
└────────────────────────────┬─────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
┌──────────────────────┐     ┌──────────────────────────┐
│    REST API           │     │    Streamlit Dashboard    │
│  (FastAPI + Uvicorn)  │     │  (Interactive Analytics)  │
│                       │     │                           │
│  POST /predict        │     │  🔮 Single Predictions   │
│  POST /predict/batch  │     │  📈 Data Analytics       │
│  GET  /health         │     │  ℹ️  Model Comparison    │
│  GET  /model/info     │     │                           │
└──────────────────────┘     └──────────────────────────┘
```

---

## 📁 Project Structure

```
churn-predictor/
├── src/
│   ├── __init__.py
│   └── pipeline.py          # Core ML pipeline (train, evaluate, predict)
├── api/
│   ├── __init__.py
│   └── app.py               # FastAPI REST endpoints
├── streamlit_app/
│   └── dashboard.py         # Interactive prediction dashboard
├── tests/
│   └── test_pipeline.py     # Unit tests
├── data/                    # Generated datasets (gitignored)
├── models/                  # Trained model artifacts (gitignored)
├── notebooks/               # Jupyter exploration notebooks
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔬 ML Pipeline

### Feature Engineering (5 custom features)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `tenure_bucket` | Categorized tenure (0-6m, 6-12m, 1-2y, 2-4y, 4-6y) | Captures non-linear tenure effects |
| `charges_per_tenure` | TotalCharges / tenure | Revenue consistency indicator |
| `high_value` | MonthlyCharges > 75th percentile | High-value customer flag |
| `contract_risk` | Is month-to-month? (binary) | Strongest churn predictor |
| `service_count` | Number of active services | Stickiness proxy |

### Models Compared

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.642 | 0.480 | 0.695 | **0.567** | **0.698** |
| Random Forest | 0.635 | 0.455 | 0.624 | 0.520 | 0.682 |
| Gradient Boosting | 0.658 | 0.525 | 0.341 | 0.414 | 0.677 |

> **Best Model:** Logistic Regression — chosen for best F1 score and recall balance (catching more churners is critical).

---

## 🌐 API Endpoints

### `POST /predict` — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.50,
    "TotalCharges": 191.00
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.747,
  "risk_level": "High",
  "contributing_factors": [
    "Month-to-month contract (high churn risk)",
    "Short tenure (2 months)",
    "High monthly charges ($95.5)",
    "Fiber optic service (higher churn segment)",
    "Electronic check payment (correlated with churn)"
  ]
}
```

### `POST /predict/batch` — Batch Predictions
### `GET /health` — Health Check
### `GET /model/info` — Model Metadata & Metrics

> **Interactive docs:** http://localhost:8000/docs (Swagger UI)

---

## 📊 Dashboard

The Streamlit dashboard provides three views:

**🔮 Predict** — Input customer details via sliders and dropdowns, get instant churn probability with a gauge chart and risk factor breakdown.

**📈 Analytics** — Explore the training dataset with interactive charts: churn rates by contract type, internet service, payment method; distribution overlays for tenure and charges.

**ℹ️ Model Info** — Compare all trained models via a radar chart (accuracy, precision, recall, F1, AUC) and view detailed metrics.

---

## 🐳 Docker

```bash
# Build & run everything
docker-compose up --build

# Or run individually
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor                                    # API only
docker run -p 8501:8501 churn-predictor streamlit run streamlit_app/dashboard.py  # Dashboard
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **ML:** Scikit-learn, Pandas, NumPy
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit, Plotly
- **DevOps:** Docker, Docker Compose
- **Testing:** Pytest

---

## 📈 Future Improvements

- [ ] Add XGBoost and LightGBM models
- [ ] Implement SHAP explainability
- [ ] Add MLflow experiment tracking
- [ ] Deploy to AWS/GCP with CI/CD
- [ ] Connect to real telecom dataset (Kaggle)
- [ ] Add model retraining scheduler
- [ ] Prometheus + Grafana monitoring

---

## 👤 Author

**Lokesh Reddy Elluri**
- MS Data Science, Indiana University Bloomington
- [LinkedIn](https://linkedin.com/in/lokeshelluri) · [Portfolio](#) · [Email](mailto:redfylokesh@gmail.com)

---

<div align="center">
<sub>Built with ❤️ as part of an end-to-end ML portfolio project</sub>
</div>
