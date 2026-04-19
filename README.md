# Customer Churn Prediction System

End-to-end ML pipeline with REST API and interactive dashboard. Predicts which telecom customers are likely to churn using ML classification, with risk scores and contributing factors for each prediction.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## Why

Customer churn is expensive for telecom companies. Identifying at-risk customers before they leave lets retention teams act early instead of reacting after the fact. This project is a full ML workflow (data generation, feature engineering, model comparison, serving via API and dashboard) wrapped around that problem.

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/Lokesh1566/churn-predictor.git
cd churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates data, trains, saves artifacts)
python src/pipeline.py

# 4. Launch the API
uvicorn api.app:app --reload --port 8000
# API docs at http://localhost:8000/docs

# 5. Launch the dashboard
streamlit run streamlit_app/dashboard.py
# Dashboard at http://localhost:8501
```

Or with Docker:

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Architecture

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
│  │  EDA &  │→ │   Feature    │→ │  Model Training  │   │
│  │ Cleaning│  │ Engineering  │  │  & Evaluation    │   │
│  └─────────┘  └──────────────┘  └──────────────────┘   │
│                                          │               │
│                    ┌─────────────────────┘               │
│                    ▼                                     │
│           ┌────────────────┐                             │
│           │ Model Artifacts│  (joblib + metadata)        │
│           └────────────────┘                             │
└────────────────────────────┬─────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────┐     ┌──────────────────────────┐
│    REST API          │     │    Streamlit Dashboard    │
│  (FastAPI + Uvicorn) │     │  (Interactive Analytics)  │
│                      │     │                           │
│  POST /predict       │     │  Single Predictions       │
│  POST /predict/batch │     │  Data Analytics           │
│  GET  /health        │     │  Model Comparison         │
│  GET  /model/info    │     │                           │
└──────────────────────┘     └──────────────────────────┘
```

## Project structure

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

## ML pipeline

### Feature engineering (5 custom features)

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `tenure_bucket` | Categorized tenure (0-6m, 6-12m, 1-2y, 2-4y, 4-6y) | Captures non-linear tenure effects |
| `charges_per_tenure` | TotalCharges / tenure | Revenue consistency indicator |
| `high_value` | MonthlyCharges > 75th percentile | High-value customer flag |
| `contract_risk` | Is month-to-month? (binary) | Strongest churn predictor |
| `service_count` | Number of active services | Stickiness proxy |

### Models compared

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.642 | 0.480 | 0.695 | **0.567** | **0.698** |
| Random Forest | 0.635 | 0.455 | 0.624 | 0.520 | 0.682 |
| Gradient Boosting | 0.658 | 0.525 | 0.341 | 0.414 | 0.677 |

Best model: Logistic Regression. I picked it for the best F1 score and recall balance. In churn prediction, missing a churner (false negative) is worse than flagging someone who wasn't going to leave (false positive), so recall matters more than raw accuracy.

## API endpoints

### `POST /predict` (single prediction)

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

Response:

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

### Other endpoints

* `POST /predict/batch` for batch predictions
* `GET /health` for health checks
* `GET /model/info` for model metadata and metrics

Interactive API docs (Swagger UI): http://localhost:8000/docs

## Dashboard

The Streamlit dashboard has three views.

**Predict.** Input customer details via sliders and dropdowns, get instant churn probability with a gauge chart and risk factor breakdown.

**Analytics.** Explore the training dataset with interactive charts. Churn rates by contract type, internet service, payment method, and distribution overlays for tenure and charges.

**Model Info.** Compare all trained models via a radar chart (accuracy, precision, recall, F1, AUC) and view detailed metrics.

## Docker

```bash
# Build and run everything
docker-compose up --build

# Or run individually
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
docker run -p 8501:8501 churn-predictor streamlit run streamlit_app/dashboard.py
```

## Testing

```bash
python -m pytest tests/ -v
```

## Tech stack

* **ML:** Scikit-learn, Pandas, NumPy
* **API:** FastAPI, Uvicorn, Pydantic
* **Dashboard:** Streamlit, Plotly
* **DevOps:** Docker, Docker Compose
* **Testing:** Pytest

## What I'd improve next

- [ ] Add XGBoost and LightGBM models
- [ ] Implement SHAP explainability
- [ ] Add MLflow experiment tracking
- [ ] Deploy to AWS/GCP with CI/CD
- [ ] Connect to a real telecom dataset (Kaggle)
- [ ] Add a model retraining scheduler
- [ ] Prometheus + Grafana monitoring

---

**Lokesh Reddy Elluri**, MS Data Science, Indiana University Bloomington  
[LinkedIn](https://linkedin.com/in/lokeshelluri) · [Email](mailto:redfylokesh@gmail.com)
