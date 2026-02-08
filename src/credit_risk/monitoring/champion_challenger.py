"""Champion-challenger evaluation on a common out-of-sample population.

A challenger replaces the production champion only when it is not materially
worse at ranking or calibration and is better on the business objective, with
uncertainty quantified by paired tests on the same applicants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.config import CostConfig
from credit_risk.evaluation.delong import delong_test
from credit_risk.features.engineering import add_engineered_features
from credit_risk.models.persistence import ModelBundle


@dataclass(frozen=True)
class PromotionRule:
    """Thresholds of the promotion decision.

    Parameters
    ----------
    auc_margin : float
        Non-inferiority margin on ROC-AUC. The lower confidence bound of
        ``AUC_challenger - AUC_champion`` must exceed ``-auc_margin``.
    brier_margin : float
        Non-inferiority margin on the Brier score. The upper confidence bound
        of ``Brier_challenger - Brier_champion`` must stay below it.
    min_profit_gain : float
        The lower confidence bound of the profit difference per applicant must
        exceed this value.
    alpha : float
        Two-sided level of every confidence interval.
    n_boot : int
        Paired bootstrap replicates for the Brier and profit intervals.
    seed : int
        Seed of the bootstrap.
    """

    auc_margin: float = 0.005
    brier_margin: float = 0.002
    min_profit_gain: float = 0.0
    alpha: float = 0.05
    n_boot: int = 1000
    seed: int = 0


@dataclass
class ChallengeResult:
    metrics: pd.DataFrame
    differences: pd.DataFrame
    swap_table: pd.DataFrame
    promote: bool
    rationale: list[str] = field(default_factory=list)


def _bootstrap_mean_ci(
    d: np.ndarray, n_boot: int, alpha: float, seed: int
) -> tuple[float, float]:
    """Percentile interval of the mean of paired per-applicant differences."""
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 100):
        stop = min(start + 100, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def swap_set_table(
    y_true, approve_champion: np.ndarray, approve_challenger: np.ndarray
) -> pd.DataFrame:
    """Cross-tabulation of the two decision policies with observed bad rates.

    ``swap_in`` are applicants the challenger approves and the champion
    declines; ``swap_out`` the reverse. A better challenger swaps in
    applicants with a lower bad rate than those it swaps out.
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(approve_champion, dtype=bool)
    b = np.asarray(approve_challenger, dtype=bool)
    cells = {
        "both_approve": a & b,
        "swap_out": a & ~b,
        "swap_in": ~a & b,
        "both_decline": ~a & ~b,
    }
    rows = [
        {
            "segment": name,
            "count": int(mask.sum()),
            "share": float(mask.mean()),
            "bad_rate": float(y[mask].mean()) if mask.any() else float("nan"),
        }
        for name, mask in cells.items()
    ]
    return pd.DataFrame(rows)


def _gain(y: np.ndarray, cost: CostConfig) -> np.ndarray:
    return np.where(y == 1, -cost.lgd, cost.margin) * cost.ead


def champion_challenger(
    y_true,
    pd_champion,
    pd_challenger,
    threshold_champion: float,
    threshold_challenger: float,
    cost: CostConfig | None = None,
    rule: PromotionRule | None = None,
) -> ChallengeResult:
    """Paired comparison of two PD models and their decision policies.

    Each model is evaluated at its own approval cut-off (approve when
    ``PD < threshold``). Discrimination is compared with the DeLong test,
    and the Brier score and realized profit per applicant with a paired
    bootstrap over applicants.
    """
    cost = cost or CostConfig()
    rule = rule or PromotionRule()
    y = np.asarray(y_true, dtype=float)
    p0 = np.asarray(pd_champion, dtype=float)
    p1 = np.asarray(pd_challenger, dtype=float)
    if not (len(y) == len(p0) == len(p1)):
        raise ValueError("y_true and both PD vectors must have the same length")

    approve0 = p0 < threshold_champion
    approve1 = p1 < threshold_challenger
    gain = _gain(y, cost)
    profit0, profit1 = gain * approve0, gain * approve1
    brier0, brier1 = (p0 - y) ** 2, (p1 - y) ** 2

    metrics = pd.DataFrame(
        [
            {
                "role": role,
                "roc_auc": float(roc_auc_score(y, p)),
                "brier": float(b.mean()),
                "threshold": float(t),
                "approval_rate": float(app.mean()),
                "bad_rate_approved": float(y[app].mean()) if app.any() else 0.0,
                "profit_per_applicant": float(pr.mean()),
            }
            for role, p, b, t, app, pr in [
                ("champion", p0, brier0, threshold_champion, approve0, profit0),
                ("challenger", p1, brier1, threshold_challenger, approve1, profit1),
            ]
        ]
    )

    dl = delong_test(y, p1, p0, alpha=rule.alpha)
    d_brier, d_profit = brier1 - brier0, profit1 - profit0
    brier_ci = _bootstrap_mean_ci(d_brier, rule.n_boot, rule.alpha, rule.seed)
    profit_ci = _bootstrap_mean_ci(d_profit, rule.n_boot, rule.alpha, rule.seed + 1)
    differences = pd.DataFrame(
        [
            {
                "metric": "roc_auc",
                "diff": dl.diff,
                "ci_low": dl.ci_low,
                "ci_high": dl.ci_high,
                "p_value": dl.p_value,
            },
            {
                "metric": "brier",
                "diff": float(d_brier.mean()),
                "ci_low": brier_ci[0],
                "ci_high": brier_ci[1],
                "p_value": float("nan"),
            },
            {
                "metric": "profit_per_applicant",
                "diff": float(d_profit.mean()),
                "ci_low": profit_ci[0],
                "ci_high": profit_ci[1],
                "p_value": float("nan"),
            },
        ]
    )

    checks = [
        (
            dl.ci_low > -rule.auc_margin,
            f"AUC difference lower bound {dl.ci_low:.4f} vs margin "
            f"{-rule.auc_margin:.4f}",
        ),
        (
            brier_ci[1] < rule.brier_margin,
            f"Brier difference upper bound {brier_ci[1]:.4f} vs margin "
            f"{rule.brier_margin:.4f}",
        ),
        (
            profit_ci[0] > rule.min_profit_gain,
            f"profit difference lower bound {profit_ci[0]:.4f} vs required "
            f"{rule.min_profit_gain:.4f}",
        ),
    ]
    rationale = [("pass: " if ok else "fail: ") + msg for ok, msg in checks]
    return ChallengeResult(
        metrics=metrics,
        differences=differences,
        swap_table=swap_set_table(y, approve0, approve1),
        promote=all(ok for ok, _ in checks),
        rationale=rationale,
    )


def challenge_bundles(
    champion: ModelBundle,
    challenger: ModelBundle,
    raw: pd.DataFrame,
    y_true,
    cost: CostConfig | None = None,
    rule: PromotionRule | None = None,
) -> ChallengeResult:
    """Score the same raw applicants with two saved bundles and compare them."""
    fe = add_engineered_features(raw)
    p0 = champion.model.predict_proba(fe[champion.features])[:, 1]
    p1 = challenger.model.predict_proba(fe[challenger.features])[:, 1]
    return champion_challenger(
        y_true, p0, p1, champion.threshold, challenger.threshold, cost, rule
    )


def report_markdown(result: ChallengeResult) -> str:
    """Plain Markdown report of a champion-challenger comparison."""

    def table(df: pd.DataFrame) -> str:
        head = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = [
            "| "
            + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
            + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([head, sep, *body])

    decision = "promote challenger" if result.promote else "retain champion"
    parts = [
        "## Champion-challenger report",
        f"Decision: {decision}",
        "### Metrics",
        table(result.metrics),
        "### Paired differences (challenger minus champion)",
        table(result.differences),
        "### Swap sets",
        table(result.swap_table),
        "### Promotion checks",
        "\n".join(f"- {line}" for line in result.rationale),
    ]
    return "\n\n".join(parts) + "\n"
