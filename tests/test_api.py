import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from credit_risk.features.engineering import FEATURE_COLUMNS, build_xy  # noqa: E402
from credit_risk.models.persistence import ModelBundle  # noqa: E402
from credit_risk.models.train import build_pipeline  # noqa: E402

APPLICANT = {
    "rev_util": 0.95,
    "age": 29,
    "late_30_59": 3,
    "debt_ratio": 0.8,
    "monthly_inc": 2500,
    "open_credit": 12,
    "late_90": 2,
    "real_estate": 0,
    "late_60_89": 1,
    "dependents": 2,
}


@pytest.fixture(scope="module")
def client(credit_df):
    from credit_risk.serving.api import create_app

    X, y = build_xy(credit_df)
    bundle = ModelBundle(
        model=build_pipeline().fit(X, y),
        features=list(FEATURE_COLUMNS),
        threshold=0.5,
        background=X.sample(300, random_state=0),
    )
    return TestClient(create_app(bundle))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_high_risk_applicant(client):
    body = client.post("/score", json=APPLICANT).json()
    assert body["decision"] == "decline"
    assert body["reasons"][0]["code"] == "DELINQ"


def test_validation_rejects_negative_income(client):
    bad = {**APPLICANT, "monthly_inc": -1}
    assert client.post("/score", json=bad).status_code == 422


def test_batch(client):
    low_risk = {
        **APPLICANT,
        "late_30_59": 0,
        "late_90": 0,
        "late_60_89": 0,
        "rev_util": 0.05,
        "age": 60,
        "monthly_inc": 9000,
    }
    body = client.post(
        "/score/batch", json={"applicants": [APPLICANT, low_risk]}
    ).json()
    pds = [r["pd"] for r in body["results"]]
    assert pds[0] > pds[1]
