"""Workforce analytics — headcount, diversity, and trend analysis."""

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class WorkforceInsight:
    total_employees: int
    avg_salary: float
    avg_satisfaction: float
    high_risk_count: int
    departments: dict
    recommendations: list

class WorkforceAnalyzer:
    def analyze(self, employees: list[dict]) -> WorkforceInsight:
        if not employees:
            return WorkforceInsight(0, 0, 0, 0, {}, [])

        df = pd.DataFrame(employees)
        recommendations = []

        avg_sat = df["satisfaction_score"].mean() if "satisfaction_score" in df else 0
        if avg_sat < 3.0:
            recommendations.append("⚠️ Low avg satisfaction — consider engagement programs")

        avg_promo = df["last_promotion_years"].mean() if "last_promotion_years" in df else 0
        if avg_promo > 3:
            recommendations.append("⚠️ Long avg time since promotion — review career ladders")

        high_risk = df[df["satisfaction_score"] < 2.5].shape[0] if "satisfaction_score" in df else 0
        if high_risk > 0:
            recommendations.append(f"🔴 {high_risk} employees at high attrition risk")

        departments = {}
        if "department" in df:
            departments = df.groupby("department").size().to_dict()

        return WorkforceInsight(
            total_employees=len(employees),
            avg_salary=round(df["salary"].mean(), 2) if "salary" in df else 0,
            avg_satisfaction=round(avg_sat, 2),
            high_risk_count=high_risk,
            departments=departments,
            recommendations=recommendations,
        )