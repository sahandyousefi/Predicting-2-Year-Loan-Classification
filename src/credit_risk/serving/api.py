from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI

from credit_risk import __version__
from credit_risk.explain.reason_codes import reason_codes
from credit_risk.explain.shap_explain import explain
from credit_risk.features.engineering import add_engineered_features
from credit_risk.models.persistence import ModelBundle, load_bundle
from credit_risk.serving.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    Reason,
    ScoreResponse,
)


def _score(bundle: ModelBundle, applicants: list[Applicant]) -> list[ScoreResponse]:
    raw = pd.DataFrame([a.model_dump() for a in applicants]).astype(float)
    X = add_engineered_features(raw)[bundle.features]
    pd_hat = bundle.model.predict_proba(X)[:, 1]
    all_reasons = reason_codes(explain(bundle.model, X, bundle.background))
    results = []
    for p, reasons in zip(pd_hat, all_reasons):
        decision = "approve" if p < bundle.threshold else "decline"
        results.append(
            ScoreResponse(
                pd=round(float(p), 6),
                decision=decision,
                threshold=round(bundle.threshold, 6),
                reasons=[Reason(**r.to_dict()) for r in reasons]
                if decision == "decline"
                else [],
            )
        )
    return results


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """Build the scoring service. Without a bundle, it is loaded from
    ``CREDIT_RISK_MODEL`` (default ``artifacts/model.joblib``)."""
    bundle = bundle or load_bundle(
        os.environ.get("CREDIT_RISK_MODEL", "artifacts/model.joblib")
    )
    app = FastAPI(title="Credit Risk Scoring", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, object]:
        return {
            "package_version": bundle.package_version,
            "threshold": bundle.threshold,
            "features": bundle.features,
            **bundle.metadata,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(applicant: Applicant) -> ScoreResponse:
        return _score(bundle, [applicant])[0]

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(request: BatchRequest) -> BatchResponse:
        return BatchResponse(results=_score(bundle, request.applicants))

    return app
