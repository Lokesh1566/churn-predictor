"""
Churn Prediction REST API
==========================
FastAPI endpoints for real-time customer churn predictions.

Run:  uvicorn api.app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib, json, os
import numpy as np
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time ML predictions for customer churn risk",
    version="1.0.0",
    contact={"name": "Lokesh Reddy Elluri", "email": "redfylokesh@gmail.com"},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load artifacts ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_LOADED = False

try:
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        model_metadata = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"Model not loaded: {e}\nRun `python src/pipeline.py` first.")


# ── Schemas ────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12, ge=0, le=72)
    PhoneService: str = Field(..., example="Yes")
    InternetService: str = Field(..., example="Fiber optic")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=79.85, ge=0)
    TotalCharges: float = Field(..., example=958.20, ge=0)

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    contributing_factors: List[str]

class BatchInput(BaseModel):
    customers: List[CustomerInput]


# ── Helpers ────────────────────────────────────────────────────
def prepare_features(data: dict):
    df = pd.DataFrame([data])
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"]
    ).astype(str)
    df["charges_per_tenure"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["high_value"] = (df["MonthlyCharges"] > 70).astype(int)
    df["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)
    df["service_count"] = (df["PhoneService"] == "Yes").astype(int) + \
                          (df["InternetService"] != "No").astype(int)
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0
    return scaler.transform(df[feature_names])

def get_risk_factors(data: dict):
    factors = []
    if data.get("Contract") == "Month-to-month":
        factors.append("Month-to-month contract (high churn risk)")
    if data.get("tenure", 99) < 12:
        factors.append(f"Short tenure ({data['tenure']} months)")
    if data.get("MonthlyCharges", 0) > 70:
        factors.append(f"High monthly charges (${data['MonthlyCharges']})")
    if data.get("InternetService") == "Fiber optic":
        factors.append("Fiber optic service (higher churn segment)")
    if data.get("PaymentMethod") == "Electronic check":
        factors.append("Electronic check payment (correlated with churn)")
    return factors[:5]


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy" if MODEL_LOADED else "model_not_loaded", "model_loaded": MODEL_LOADED}

@app.get("/model/info")
def model_info():
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded. Run pipeline.py first.")
    return model_metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    data = customer.dict()
    features = prepare_features(data)
    prob = float(model.predict_proba(features)[0][1])
    return {
        "churn_prediction": int(prob >= 0.5),
        "churn_probability": round(prob, 4),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
        "contributing_factors": get_risk_factors(data),
    }

@app.post("/predict/batch")
def predict_batch(batch: BatchInput):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    results = []
    for c in batch.customers:
        data = c.dict()
        prob = float(model.predict_proba(prepare_features(data))[0][1])
        results.append({
            "churn_prediction": int(prob >= 0.5),
            "churn_probability": round(prob, 4),
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
            "contributing_factors": get_risk_factors(data),
        })
    return {
        "predictions": results,
        "summary": {
            "total": len(results),
            "predicted_churners": sum(r["churn_prediction"] for r in results),
            "avg_churn_probability": round(np.mean([r["churn_probability"] for r in results]), 4),
        }
    }
