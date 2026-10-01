"""Contract, behaviour, validation, and batching tests for the Lab 1 API."""

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app


GOOD_APPLICANT = {
    "limit_bal": 300000,
    "sex": 2,
    "education": 1,
    "marriage": 2,
    "age": 38,
    "pay_status": [-1, -1, -1, -1, -1, -1],
    "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
    "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500],
}

RISKY_APPLICANT = {
    "limit_bal": 20000,
    "sex": 1,
    "education": 3,
    "marriage": 1,
    "age": 24,
    "pay_status": [4, 3, 3, 2, 2, 2],
    "bill_amt": [19800, 19500, 19000, 18500, 18000, 17500],
    "pay_amt": [0, 0, 200, 0, 300, 0],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_model_info(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.0.0",
    }
    info = client.get("/model/info").json()
    assert info["is_loaded"] is True
    assert 0.5 < info["metrics"]["roc_auc"] <= 1.0


def test_predict_contract_and_determinism(client):
    first = client.post("/predict", json=GOOD_APPLICANT)
    second = client.post("/predict", json=GOOD_APPLICANT)
    assert first.status_code == 200
    assert first.json() == second.json()
    data = first.json()
    assert set(data) == {
        "default_probability",
        "risk_band",
        "decision",
        "review_threshold",
        "decline_threshold",
        "model_version",
    }
    assert 0.0 <= data["default_probability"] <= 1.0


@pytest.mark.parametrize("applicant", [GOOD_APPLICANT, RISKY_APPLICANT])
def test_decision_matches_thresholds(client, applicant):
    data = client.post("/predict", json=applicant).json()
    probability = data["default_probability"]
    if probability >= data["decline_threshold"]:
        assert (data["risk_band"], data["decision"]) == ("HIGH", "DECLINE")
    elif probability >= data["review_threshold"]:
        assert (data["risk_band"], data["decision"]) == ("MEDIUM", "REVIEW")
    else:
        assert (data["risk_band"], data["decision"]) == ("LOW", "APPROVE")


def test_model_behaviour(client):
    good = client.post("/predict", json=GOOD_APPLICANT).json()["default_probability"]
    risky = client.post("/predict", json=RISKY_APPLICANT).json()["default_probability"]
    assert risky > good

    mild = copy.deepcopy(GOOD_APPLICANT)
    severe = copy.deepcopy(GOOD_APPLICANT)
    mild["pay_status"] = [1, 0, 0, 0, 0, 0]
    severe["pay_status"] = [4, 3, 3, 2, 2, 2]
    mild_score = client.post("/predict", json=mild).json()["default_probability"]
    severe_score = client.post("/predict", json=severe).json()["default_probability"]
    assert severe_score >= mild_score


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sex", 3), ("education", 9), ("marriage", 0),
        ("age", 12), ("age", 150), ("limit_bal", 0), ("limit_bal", -5000),
    ],
)
def test_out_of_range_values_are_rejected(client, field, value):
    payload = copy.deepcopy(GOOD_APPLICANT)
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.pop("age"),
        lambda p: p.update(pay_status=[0, 0, 0]),
        lambda p: p.update(pay_status=[99, 0, 0, 0, 0, 0]),
        lambda p: p.update(pay_amt=[-1, 0, 0, 0, 0, 0]),
    ],
)
def test_invalid_payloads_are_rejected(client, mutator):
    payload = copy.deepcopy(GOOD_APPLICANT)
    mutator(payload)
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_batch_preserves_order_and_matches_single(client):
    response = client.post(
        "/predict/batch",
        json={"applications": [GOOD_APPLICANT, RISKY_APPLICANT, GOOD_APPLICANT]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == len(data["predictions"]) == 3
    assert data["predictions"][1]["default_probability"] > data["predictions"][0]["default_probability"]
    assert data["predictions"][0] == client.post("/predict", json=GOOD_APPLICANT).json()


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422
