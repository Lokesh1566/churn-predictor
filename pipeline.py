"""
Customer Churn Prediction Pipeline
===================================
End-to-end ML pipeline: data generation → preprocessing → feature engineering
→ model training → evaluation → model persistence

Author: Lokesh Reddy Elluri
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")


class ChurnPredictor:
    """End-to-end customer churn prediction pipeline."""

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42,
                class_weight="balanced", n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
            ),
        }
        self.results = {}

    # ── Data Generation ────────────────────────────────────────
    def generate_synthetic_data(self, n_samples=5000):
        """Generate realistic synthetic telecom churn data."""
        np.random.seed(42)
        data = {
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "gender": np.random.choice(["Male", "Female"], n_samples),
            "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
            "Partner": np.random.choice(["Yes", "No"], n_samples, p=[0.52, 0.48]),
            "Dependents": np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70]),
            "tenure": np.clip(np.random.exponential(32, n_samples).astype(int), 0, 72),
            "PhoneService": np.random.choice(["Yes", "No"], n_samples, p=[0.90, 0.10]),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
            ),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]
            ),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples, p=[0.60, 0.40]),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                n_samples, p=[0.34, 0.23, 0.22, 0.21]
            ),
            "MonthlyCharges": np.round(np.random.uniform(18, 118, n_samples), 2),
        }
        df = pd.DataFrame(data)
        df["TotalCharges"] = np.round(
            df["tenure"] * df["MonthlyCharges"] * np.random.uniform(0.85, 1.15, n_samples), 2
        )

        # Realistic churn probability based on features
        churn_prob = np.zeros(n_samples)
        churn_prob += (df["Contract"] == "Month-to-month").astype(float) * 0.25
        churn_prob += (df["InternetService"] == "Fiber optic").astype(float) * 0.10
        churn_prob += (df["PaymentMethod"] == "Electronic check").astype(float) * 0.08
        churn_prob += (df["tenure"] < 12).astype(float) * 0.15
        churn_prob += (df["MonthlyCharges"] > 70).astype(float) * 0.08
        churn_prob += (df["SeniorCitizen"] == 1).astype(float) * 0.05
        churn_prob += (df["Partner"] == "No").astype(float) * 0.04
        churn_prob += (df["PaperlessBilling"] == "Yes").astype(float) * 0.03
        churn_prob = np.clip(churn_prob + np.random.normal(0, 0.08, n_samples), 0.02, 0.95)
        df["Churn"] = (np.random.random(n_samples) < churn_prob).astype(int)

        return df

    # ── Data Loading ───────────────────────────────────────────
    def load_data(self):
        if self.data_path and os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
            print(f"[+] Loaded {len(df)} records from {self.data_path}")
        else:
            df = self.generate_synthetic_data()
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/telecom_churn.csv", index=False)
            print(f"[+] Generated {len(df)} synthetic records -> data/telecom_churn.csv")
        return df

    # ── EDA ────────────────────────────────────────────────────
    def explore_data(self, df):
        stats = {
            "total_records": len(df),
            "churn_rate": round(df["Churn"].mean() * 100, 2),
            "avg_tenure": round(df["tenure"].mean(), 1),
            "avg_monthly_charges": round(df["MonthlyCharges"].mean(), 2),
            "churn_by_contract": df.groupby("Contract")["Churn"].mean().round(3).to_dict(),
            "churn_by_internet": df.groupby("InternetService")["Churn"].mean().round(3).to_dict(),
        }
        print(f"\n{'='*50}")
        print("  EXPLORATORY DATA ANALYSIS")
        print(f"{'='*50}")
        print(f"  Total Records:    {stats['total_records']}")
        print(f"  Churn Rate:       {stats['churn_rate']}%")
        print(f"  Avg Tenure:       {stats['avg_tenure']} months")
        print(f"  Avg Monthly Bill: ${stats['avg_monthly_charges']}")
        print(f"\n  Churn by Contract Type:")
        for k, v in stats["churn_by_contract"].items():
            print(f"    {k}: {v*100:.1f}%")
        return stats

    # ── Preprocessing & Feature Engineering ────────────────────
    def preprocess(self, df):
        df = df.copy()
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        # Feature engineering
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"]
        ).astype(str)
        df["charges_per_tenure"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
        )
        df["high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)).astype(int)
        df["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)
        df["service_count"] = (
            (df["PhoneService"] == "Yes").astype(int) +
            (df["InternetService"] != "No").astype(int)
        )

        # Encode categoricals
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop("Churn", axis=1)
        y = df["Churn"]
        self.feature_names = X.columns.tolist()

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        print(f"[+] Preprocessed {len(X.columns)} features (including 5 engineered)")
        return X_scaled, y

    # ── Training & Evaluation ──────────────────────────────────
    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\n{'='*50}")
        print("  MODEL TRAINING & EVALUATION")
        print(f"{'='*50}")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Churn ratio: {y_train.mean()*100:.1f}%\n")

        best_f1 = 0
        for name, model in self.models.items():
            print(f"  Training {name}...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1_score": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
                "cv_f1_mean": round(cv_scores.mean(), 4),
                "cv_f1_std": round(cv_scores.std(), 4),
            }
            self.results[name] = metrics
            print(f"    Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | "
                  f"AUC: {metrics['roc_auc']:.4f} | CV-F1: {metrics['cv_f1_mean']:.4f}")

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                self.best_model = model
                self.best_model_name = name

        print(f"\n  Best Model: {self.best_model_name} (F1: {best_f1:.4f})")

        if hasattr(self.best_model, "feature_importances_"):
            imp = pd.Series(self.best_model.feature_importances_, index=self.feature_names)
            imp = imp.sort_values(ascending=False)
            print(f"\n  Top 10 Features:")
            for feat, val in imp.head(10).items():
                bar = "█" * int(val * 80)
                print(f"    {feat:25s} {val:.4f} {bar}")

        return self.results

    # ── Save Artifacts ─────────────────────────────────────────
    def save_model(self, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(output_dir, "best_model.joblib"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.joblib"))
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.joblib"))

        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f)

        metadata = {
            "model_name": self.best_model_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": self.results.get(self.best_model_name, {}),
            "all_results": self.results,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[+] Model artifacts saved to {output_dir}/")
        return os.path.join(output_dir, "best_model.joblib")

    # ── Single Prediction ──────────────────────────────────────
    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        df["tenure_bucket"] = pd.cut(
            df["tenure"], bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"]
        ).astype(str)
        df["charges_per_tenure"] = np.where(
            df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
        )
        df["high_value"] = (df["MonthlyCharges"] > 70).astype(int)
        df["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)
        df["service_count"] = (
            (df["PhoneService"] == "Yes").astype(int) +
            (df["InternetService"] != "No").astype(int)
        )
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        df = df[self.feature_names]
        scaled = self.scaler.transform(df)
        prob = self.best_model.predict_proba(scaled)[0][1]
        return {
            "churn_prediction": int(prob >= 0.5),
            "churn_probability": round(float(prob), 4),
            "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low",
            "model_used": self.best_model_name,
        }


def run_pipeline():
    print("=" * 50)
    print("  CUSTOMER CHURN PREDICTION PIPELINE")
    print("=" * 50)

    pipe = ChurnPredictor()
    df = pipe.load_data()
    pipe.explore_data(df)
    X, y = pipe.preprocess(df)
    pipe.train_and_evaluate(X, y)
    pipe.save_model()

    # Test prediction
    print(f"\n{'='*50}")
    print("  SAMPLE PREDICTION")
    print(f"{'='*50}")
    sample = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
        "InternetService": "Fiber optic", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50, "TotalCharges": 191.00,
    }
    result = pipe.predict(sample)
    print(f"  Churn Probability: {result['churn_probability']*100:.1f}%")
    print(f"  Risk Level:        {result['risk_level']}")
    print(f"  Prediction:        {'WILL CHURN' if result['churn_prediction'] else 'WILL STAY'}")

    return pipe


if __name__ == "__main__":
    run_pipeline()
