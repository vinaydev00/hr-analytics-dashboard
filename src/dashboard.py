"""
Streamlit HR Analytics Dashboard
Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.attrition_model import AttritionModel

st.set_page_config(page_title="HR Analytics Dashboard", page_icon="👥", layout="wide")

@st.cache_resource
def load_model():
    return AttritionModel()

model = load_model()

st.title("👥 HR Analytics Dashboard")
st.markdown("ML-powered employee attrition prediction and workforce analytics")

# Sidebar
st.sidebar.header("Employee Profile")
age = st.sidebar.slider("Age", 22, 60, 35)
years = st.sidebar.slider("Years at Company", 0, 20, 3)
salary = st.sidebar.slider("Salary ($)", 30000, 150000, 70000)
satisfaction = st.sidebar.slider("Satisfaction Score", 1.0, 5.0, 3.5)
wlb = st.sidebar.slider("Work-Life Balance", 1.0, 5.0, 3.5)
promotion = st.sidebar.slider("Years Since Promotion", 0, 10, 2)
projects = st.sidebar.slider("Number of Projects", 1, 10, 4)

employee = {
    "id": "EMP001", "age": age, "years_at_company": years,
    "salary": salary, "satisfaction_score": satisfaction,
    "work_life_balance": wlb, "last_promotion_years": promotion,
    "num_projects": projects,
}

prediction = model.predict(employee)

col1, col2, col3 = st.columns(3)
col1.metric("Attrition Risk", f"{prediction.attrition_risk:.0%}")
col2.metric("Risk Level", prediction.risk_level)
col3.metric("Top Factor", prediction.top_factors[0].replace("_", " ").title())

# Risk gauge
fig = px.bar(
    x=["Attrition Risk"], y=[prediction.attrition_risk],
    color=[prediction.risk_level],
    color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
    title="Attrition Risk Score",
)
st.plotly_chart(fig, use_container_width=True)

# Feature importance
st.subheader("Key Risk Factors")
importance_df = pd.DataFrame(
    list(model.feature_importances_.items()),
    columns=["Factor", "Importance"]
).sort_values("Importance", ascending=True)
fig2 = px.bar(importance_df, x="Importance", y="Factor", orientation="h", title="Feature Importance")
st.plotly_chart(fig2, use_container_width=True)