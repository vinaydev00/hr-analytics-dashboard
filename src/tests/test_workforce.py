from src.workforce_analyzer import WorkforceAnalyzer

def test_analyze_empty():
    a = WorkforceAnalyzer()
    result = a.analyze([])
    assert result.total_employees == 0

def test_analyze_count():
    a = WorkforceAnalyzer()
    employees = [
        {"salary": 70000, "satisfaction_score": 4.0, "last_promotion_years": 1},
        {"salary": 80000, "satisfaction_score": 2.0, "last_promotion_years": 6},
    ]
    result = a.analyze(employees)
    assert result.total_employees == 2

def test_high_risk_detected():
    a = WorkforceAnalyzer()
    employees = [{"salary": 50000, "satisfaction_score": 1.5, "last_promotion_years": 7}]
    result = a.analyze(employees)
    assert result.high_risk_count == 1

def test_recommendations_generated():
    a = WorkforceAnalyzer()
    employees = [{"salary": 50000, "satisfaction_score": 2.0, "last_promotion_years": 5}]
    result = a.analyze(employees)
    assert len(result.recommendations) > 0