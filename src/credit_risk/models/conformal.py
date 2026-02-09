"""Distribution-free uncertainty for PD estimates.

Two methods are provided, both fitted on a held-out calibration sample of
model scores and labels and both valid under exchangeability alone.

Mondrian (label-conditional) conformal prediction returns, for each
applicant, the set of outcomes that cannot be excluded at level ``alpha``.
Coverage holds within each class, so it is not dominated by the majority
class. A singleton set supports an automatic decision; a set with both
outcomes marks an applicant the model cannot separate at that level.

Inductive Venn-Abers prediction returns an interval ``[p0, p1]`` for the
probability of default. One of the two bounds is a calibrated probability in
the sense of Venn predictors; the width of the interval measures how much the
calibration sample constrains the estimate for that score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DECISIONS = {(True, False): "approve", (False, True): "decline"}


class MondrianConformalClassifier:
    """Label-conditional split conformal predictor for binary outcomes.

    The nonconformity of label ``c`` for an applicant with PD ``p`` is
    ``1 - p`` for ``c = 1`` and ``p`` for ``c = 0``. The p-value of label ``c``
    is the share of calibration applicants of class ``c`` that are at least as
    nonconforming, with the test point counted.
    """

    def fit(self, y_cal, p_cal) -> MondrianConformalClassifier:
        y = np.asarray(y_cal, dtype=int)
        p = np.asarray(p_cal, dtype=float)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("calibration sample needs both classes")
        # Sorted nonconformity scores per class.
        self.scores_ = {0: np.sort(p[y == 0]), 1: np.sort(1 - p[y == 1])}
        return self

    def p_values(self, p_test) -> np.ndarray:
        """Array of shape ``(n, 2)`` with the p-value of label 0 and label 1."""
        p = np.asarray(p_test, dtype=float)
        out = np.empty((len(p), 2))
        for c, a_test in ((0, p), (1, 1 - p)):
            cal = self.scores_[c]
            at_least = len(cal) - np.searchsorted(cal, a_test, side="left")
            out[:, c] = (at_least + 1) / (len(cal) + 1)
        return out

    def predict_set(self, p_test, alpha: float = 0.1) -> np.ndarray:
        """Boolean array of shape ``(n, 2)``: label ``c`` is in the set."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        return self.p_values(p_test) > alpha


def three_way_decision(sets: np.ndarray) -> np.ndarray:
    """Approve on ``{0}``, decline on ``{1}``, refer otherwise."""
    return np.array([DECISIONS.get((bool(a), bool(b)), "refer") for a, b in sets])


def set_summary(sets: np.ndarray, y_true=None) -> dict[str, float]:
    """Shares of set types and, with labels, coverage overall and by class."""
    sets = np.asarray(sets, dtype=bool)
    size = sets.sum(axis=1)
    out = {
        "share_empty": float((size == 0).mean()),
        "share_only_0": float((sets[:, 0] & ~sets[:, 1]).mean()),
        "share_only_1": float((~sets[:, 0] & sets[:, 1]).mean()),
        "share_both": float((size == 2).mean()),
        "mean_size": float(size.mean()),
    }
    if y_true is not None:
        y = np.asarray(y_true, dtype=int)
        covered = sets[np.arange(len(y)), y]
        out["coverage"] = float(covered.mean())
        out["coverage_0"] = float(covered[y == 0].mean())
        out["coverage_1"] = float(covered[y == 1].mean())
    return out


@dataclass
class VennAbersResult:
    p0: np.ndarray
    p1: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.p1 - self.p0

    @property
    def merged(self) -> np.ndarray:
        """Single probability ``p1 / (1 - p0 + p1)``, optimal for log loss."""
        return self.p1 / (1.0 - self.p0 + self.p1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"p0": self.p0, "p1": self.p1, "pd": self.merged, "width": self.width}
        )


class VennAbersCalibrator:
    """Inductive Venn-Abers predictor on model scores.

    For a test score ``s`` the calibration sample is extended with ``(s, 0)``
    and with ``(s, 1)``, an isotonic regression is fitted to each, and the two
    fitted values at ``s`` give ``p0`` and ``p1``. Isotonic regression depends
    only on the order of the scores, so the result is determined by where
    ``s`` falls among the calibration scores and is cached on that position.
    """

    def fit(self, y_cal, p_cal) -> VennAbersCalibrator:
        y = np.asarray(y_cal, dtype=float)
        s = np.asarray(p_cal, dtype=float)
        order = np.argsort(s, kind="stable")
        self.scores_, self.labels_ = s[order], y[order]
        return self

    def _fit_value(self, score: float, label: float) -> float:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.append(self.scores_, score), np.append(self.labels_, label))
        return float(iso.predict([score])[0])

    def predict_interval(self, p_test) -> VennAbersResult:
        s = np.asarray(p_test, dtype=float)
        left = np.searchsorted(self.scores_, s, side="left")
        right = np.searchsorted(self.scores_, s, side="right")
        cache: dict[tuple[int, int], tuple[float, float]] = {}
        p0 = np.empty(len(s))
        p1 = np.empty(len(s))
        for i, (score, key) in enumerate(zip(s, zip(left, right))):
            if key not in cache:
                cache[key] = (self._fit_value(score, 0.0), self._fit_value(score, 1.0))
            p0[i], p1[i] = cache[key]
        return VennAbersResult(p0=p0, p1=p1)

    def predict_proba(self, p_test) -> np.ndarray:
        return self.predict_interval(p_test).merged


def uncertainty_report(
    y_cal, p_cal, y_test, p_test, alphas: tuple[float, ...] = (0.05, 0.1, 0.2)
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Conformal set summaries for several levels and Venn-Abers statistics.

    Returns
    -------
    sets : pandas.DataFrame
        One row per ``alpha`` with set shares and coverage on the test sample.
    venn_abers : dict
        Mean and 90th percentile of the interval width, and the expected
        calibration error of the raw and the merged Venn-Abers PD.
    """
    from credit_risk.evaluation.calibration import expected_calibration_error

    cp = MondrianConformalClassifier().fit(y_cal, p_cal)
    rows = [
        {"alpha": a, **set_summary(cp.predict_set(p_test, a), y_test)} for a in alphas
    ]
    va = VennAbersCalibrator().fit(y_cal, p_cal).predict_interval(p_test)
    stats = {
        "mean_width": float(va.width.mean()),
        "p90_width": float(np.quantile(va.width, 0.9)),
        "ece_raw": expected_calibration_error(y_test, p_test),
        "ece_venn_abers": expected_calibration_error(y_test, va.merged),
    }
    return pd.DataFrame(rows), stats
