from __future__ import annotations

import pandas as pd

from credit_risk.data.schema import validate_frame
from credit_risk.explain.reason_codes import reason_codes
from credit_risk.explain.shap_explain import explain
from credit_risk.features.engineering import add_engineered_features
from credit_risk.models.persistence import ModelBundle


def score_frame(
    bundle: ModelBundle, raw: pd.DataFrame, max_reasons: int = 4
) -> pd.DataFrame:
    """PD, decision and principal reasons for each row of raw applicant data."""
    validate_frame(raw, require_target=False).raise_for_errors()
    X = add_engineered_features(raw)[bundle.features]
    pd_hat = bundle.model.predict_proba(X)[:, 1]
    reasons = reason_codes(explain(bundle.model, X, bundle.background), max_reasons)
    out = pd.DataFrame(index=raw.index)
    out["pd"] = pd_hat.round(6)
    out["decision"] = ["approve" if p < bundle.threshold else "decline" for p in pd_hat]
    out["reason_codes"] = [
        ";".join(r.code for r in rs) if d == "decline" else ""
        for rs, d in zip(reasons, out["decision"])
    ]
    return out
