"""Tests for Churn Prediction Pipeline"""
import pytest, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.pipeline import ChurnPredictor

class TestChurnPredictor:
    def setup_method(self):
        self.p = ChurnPredictor()

    def test_synthetic_data_shape(self):
        df = self.p.generate_synthetic_data(100)
        assert len(df) == 100
        assert "Churn" in df.columns
        assert df["Churn"].isin([0, 1]).all()

    def test_churn_rate_realistic(self):
        df = self.p.generate_synthetic_data(5000)
        assert 0.15 < df["Churn"].mean() < 0.50

    def test_preprocessing_no_nulls(self):
        df = self.p.generate_synthetic_data(200)
        X, y = self.p.preprocess(df)
        assert X.isnull().sum().sum() == 0
        assert len(y) == 200

    def test_feature_engineering(self):
        df = self.p.generate_synthetic_data(100)
        X, _ = self.p.preprocess(df)
        for col in ["tenure_bucket", "charges_per_tenure", "high_value", "contract_risk", "service_count"]:
            assert col in X.columns

    def test_training_produces_results(self):
        df = self.p.generate_synthetic_data(500)
        X, y = self.p.preprocess(df)
        results = self.p.train_and_evaluate(X, y)
        assert len(results) == 3
        assert self.p.best_model is not None
        for metrics in results.values():
            assert 0 <= metrics["accuracy"] <= 1
            assert 0 <= metrics["roc_auc"] <= 1

    def test_prediction_output(self):
        df = self.p.generate_synthetic_data(500)
        X, y = self.p.preprocess(df)
        self.p.train_and_evaluate(X, y)
        result = self.p.predict({
            "gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
            "tenure":2,"PhoneService":"Yes","InternetService":"Fiber optic",
            "Contract":"Month-to-month","PaperlessBilling":"Yes",
            "PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":191.0
        })
        assert 0 <= result["churn_probability"] <= 1
        assert result["risk_level"] in ["Low", "Medium", "High"]

    def test_model_save(self, tmp_path):
        df = self.p.generate_synthetic_data(300)
        X, y = self.p.preprocess(df)
        self.p.train_and_evaluate(X, y)
        self.p.save_model(str(tmp_path))
        assert os.path.exists(os.path.join(tmp_path, "best_model.joblib"))
        assert os.path.exists(os.path.join(tmp_path, "model_metadata.json"))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
