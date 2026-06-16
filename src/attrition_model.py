"""
HR Attrition Predictor
Predicts employee attrition risk using ML.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

@dataclass
class AttritionPrediction:
    employee_id: str
    attrition_risk: float
    risk_level: str
    top_factors: list

class AttritionModel:
    FEATURE_COLUMNS = [
        "age", "years_at_company", "salary", "satisfaction_score",
        "work_life_balance", "last_promotion_years", "num_projects",
    ]

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False
        self._train_on_synthetic()

    def _generate_synthetic_data(self, n: int = 500) -> pd.DataFrame:
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(22, 60, n),
            "years_at_company": np.random.randint(0, 20, n),
            "salary": np.random.randint(30000, 150000, n),
            "satisfaction_score": np.random.uniform(1, 5, n),
            "work_life_balance": np.random.uniform(1, 5, n),
            "last_promotion_years": np.random.randint(0, 10, n),
            "num_projects": np.random.randint(1, 10, n),
        })
        # Attrition more likely with low satisfaction and long time since promotion
        df["attrition"] = (
            (df["satisfaction_score"] < 2.5) |
            (df["last_promotion_years"] > 5) |
            (df["work_life_balance"] < 2.0)
        ).astype(int)
        return df

    def _train_on_synthetic(self):
        df = self._generate_synthetic_data()
        X = df[self.FEATURE_COLUMNS]
        y = df["attrition"]
        self.model.fit(X, y)
        self.trained = True
        self.feature_importances_ = dict(zip(self.FEATURE_COLUMNS, self.model.feature_importances_))

    def predict(self, employee_data: dict) -> AttritionPrediction:
        features = np.array([[employee_data.get(f, 0) for f in self.FEATURE_COLUMNS]])
        proba = self.model.predict_proba(features)[0][1]
        risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
        top_factors = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:3]
        return AttritionPrediction(
            employee_id=employee_data.get("id", "unknown"),
            attrition_risk=round(float(proba), 3),
            risk_level=risk_level,
            top_factors=[f[0] for f in top_factors],
        )

    def batch_predict(self, employees: list[dict]) -> list[AttritionPrediction]:
        return [self.predict(e) for e in employees]