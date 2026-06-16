import pytest
from src.attrition_model import AttritionModel, AttritionPrediction

@pytest.fixture
def model():
    return AttritionModel()

@pytest.fixture
def sample_employee():
    return {
        "id": "EMP001", "age": 35, "years_at_company": 3,
        "salary": 70000, "satisfaction_score": 2.0,
        "work_life_balance": 1.5, "last_promotion_years": 6,
        "num_projects": 4,
    }

def test_predict_returns_prediction(model, sample_employee):
    result = model.predict(sample_employee)
    assert isinstance(result, AttritionPrediction)

def test_risk_score_between_0_and_1(model, sample_employee):
    result = model.predict(sample_employee)
    assert 0 <= result.attrition_risk <= 1

def test_risk_level_valid(model, sample_employee):
    result = model.predict(sample_employee)
    assert result.risk_level in ["Low", "Medium", "High"]

def test_top_factors_not_empty(model, sample_employee):
    result = model.predict(sample_employee)
    assert len(result.top_factors) > 0

def test_batch_predict(model):
    employees = [
        {"id": f"EMP{i}", "age": 30+i, "years_at_company": i,
         "salary": 60000, "satisfaction_score": 3.0,
         "work_life_balance": 3.0, "last_promotion_years": 2,
         "num_projects": 3}
        for i in range(5)
    ]
    results = model.batch_predict(employees)
    assert len(results) == 5