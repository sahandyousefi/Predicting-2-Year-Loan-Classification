from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _initial_edges(x: np.ndarray, max_bins: int) -> np.ndarray:
    qs = np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1])
    return np.unique(qs)


def _bin_counts(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(edges, x, side="right")
    n_bins = len(edges) + 1
    events = np.bincount(idx, weights=y, minlength=n_bins)
    totals = np.bincount(idx, minlength=n_bins).astype(float)
    return events, totals - events


def _merge_until_monotonic(
    edges: np.ndarray, events: np.ndarray, non_events: np.ndarray
) -> np.ndarray:
    edges = list(edges)
    events, non_events = list(events), list(non_events)
    while len(edges) > 0:
        rates = np.array(events) / (np.array(events) + np.array(non_events))
        diffs = np.diff(rates)
        direction = np.sign(np.nansum(diffs)) or 1.0
        breaks = np.where(np.sign(diffs) * direction < 0)[0]
        if len(breaks) == 0:
            break
        i = int(breaks[0])
        # Merge bin i+1 into bin i by dropping the edge between them.
        events[i] += events.pop(i + 1)
        non_events[i] += non_events.pop(i + 1)
        edges.pop(i)
    return np.array(edges)


def _merge_small_bins(
    edges: np.ndarray, events: np.ndarray, non_events: np.ndarray, min_count: int
) -> np.ndarray:
    edges = list(edges)
    events, non_events = list(events), list(non_events)
    while len(edges) > 0:
        totals = np.array(events) + np.array(non_events)
        small = np.where(totals < min_count)[0]
        if len(small) == 0:
            break
        i = int(small[0])
        j = i - 1 if i == len(totals) - 1 else i
        events[j] += events.pop(j + 1)
        non_events[j] += non_events.pop(j + 1)
        edges.pop(j)
    return np.array(edges)


class WoEBinner(BaseEstimator, TransformerMixin):
    """Quantile binning with Weight of Evidence encoding.

    WoE for bin :math:`i` is :math:`\\ln(g_i / G) - \\ln(b_i / B)`, where
    :math:`g_i, b_i` are non-event and event counts in the bin and :math:`G, B`
    the totals. Information Value is
    :math:`\\sum_i (g_i / G - b_i / B)\\,\\mathrm{WoE}_i`. A smoothing constant is
    added to each count so empty cells do not produce infinite values.

    Parameters
    ----------
    max_bins : int
        Number of quantile bins before merging.
    min_bin_share : float
        Bins holding less than this share of rows are merged with a neighbour.
    monotonic : bool
        Merge adjacent bins until the event rate is monotonic in the feature.
    smoothing : float
        Pseudo-count added to event and non-event counts.
    """

    def __init__(
        self,
        max_bins: int = 10,
        min_bin_share: float = 0.05,
        monotonic: bool = True,
        smoothing: float = 0.5,
    ) -> None:
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.monotonic = monotonic
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> WoEBinner:
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float)
        min_count = int(np.ceil(self.min_bin_share * len(X)))
        self.feature_names_in_ = np.array(X.columns)
        self.edges_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, np.ndarray] = {}
        tables = []
        total_events = y.sum()
        total_non = len(y) - total_events

        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            edges = _initial_edges(x, self.max_bins)
            edges = _merge_small_bins(edges, *_bin_counts(x, y, edges), min_count)
            if self.monotonic:
                edges = _merge_until_monotonic(edges, *_bin_counts(x, y, edges))
            events, non_events = _bin_counts(x, y, edges)
            s = self.smoothing
            dist_event = (events + s) / (total_events + s * len(events))
            dist_non = (non_events + s) / (total_non + s * len(events))
            woe = np.log(dist_non / dist_event)
            self.edges_[col] = edges
            self.woe_[col] = woe

            lower = np.r_[-np.inf, edges]
            upper = np.r_[edges, np.inf]
            tables.append(
                pd.DataFrame(
                    {
                        "feature": col,
                        "bin": [
                            f"[{lo:.4g}, {hi:.4g})" for lo, hi in zip(lower, upper)
                        ],
                        "count": events + non_events,
                        "events": events,
                        "event_rate": events / np.maximum(events + non_events, 1),
                        "woe": woe,
                        "iv_contribution": (dist_non - dist_event) * woe,
                    }
                )
            )
        self.bin_table_ = pd.concat(tables, ignore_index=True)
        self.iv_ = (
            self.bin_table_.groupby("feature")["iv_contribution"]
            .sum()
            .sort_values(ascending=False)
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = {}
        for col in self.feature_names_in_:
            idx = np.searchsorted(
                self.edges_[col], X[col].to_numpy(dtype=float), side="right"
            )
            out[col] = self.woe_[col][idx]
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_in_


def iv_strength(iv: float) -> str:
    """Conventional reading of Information Value (Siddiqi, 2006)."""
    if iv < 0.02:
        return "not predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    return "strong"
