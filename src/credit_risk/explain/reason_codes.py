from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from credit_risk.explain.shap_explain import Attribution

# Engineered features are grouped back to the borrower attribute they come
# from, so a notice never cites a transformed variable.
FEATURE_GROUP = {
    "late_30_59": "DELINQ",
    "late_60_89": "DELINQ",
    "late_90": "DELINQ",
    "late_severity": "DELINQ",
    "rev_util": "UTIL",
    "log_rev_util": "UTIL",
    "debt_ratio": "DEBT",
    "log_debt_ratio": "DEBT",
    "debt_per_dependent": "DEBT",
    "monthly_inc": "INCOME",
    "log_monthly_inc": "INCOME",
    "inc_per_dependent": "INCOME",
    "credit_to_income": "EXPOSURE",
    "open_credit": "EXPOSURE",
    "real_estate": "EXPOSURE",
    "age": "HISTORY",
    "dependents": "HOUSEHOLD",
}

REASON_TEXT = {
    "DELINQ": "Number or severity of past due payments",
    "UTIL": "Proportion of revolving balances to credit limits is too high",
    "DEBT": "Debt obligations are high relative to income",
    "INCOME": "Income is insufficient for the amount of credit",
    "EXPOSURE": "Number of open credit accounts relative to income",
    "HISTORY": "Length of credit history",
    "HOUSEHOLD": "Household obligations relative to income",
}


@dataclass(frozen=True)
class ReasonCode:
    code: str
    description: str
    contribution: float

    def to_dict(self) -> dict:
        return asdict(self)


def group_attributions(attr: Attribution) -> pd.DataFrame:
    groups = attr.values.columns.map(lambda c: FEATURE_GROUP.get(c, c.upper()))
    return attr.values.T.groupby(groups).sum().T


def reason_codes(
    attr: Attribution, max_reasons: int = 4, min_contribution: float = 0.0
) -> list[list[ReasonCode]]:
    """Principal reasons for each applicant, ordered by contribution to risk.

    Only groups that push the log-odds of default up relative to the average
    applicant are eligible, which matches the purpose of an adverse action
    notice (Regulation B requires up to four principal reasons).
    """
    grouped = group_attributions(attr)
    out = []
    for _, row in grouped.iterrows():
        positive = row[row > min_contribution].sort_values(ascending=False)
        out.append(
            [
                ReasonCode(code, REASON_TEXT.get(code, code), round(float(v), 4))
                for code, v in positive.head(max_reasons).items()
            ]
        )
    return out
