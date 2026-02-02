from __future__ import annotations

import inspect

from sklearn.linear_model import LogisticRegression

_PENALTY_DEPRECATED = (
    inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"
)


def make_logistic(
    C: float = 1.0,
    l1_ratio: float = 0.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    random_state: int | None = None,
) -> LogisticRegression:
    """Build a logistic regression that works across scikit-learn versions.

    scikit-learn 1.8 deprecated ``penalty`` in favour of ``l1_ratio`` alone.
    Older versions only read ``l1_ratio`` with the elastic-net penalty.

    Parameters
    ----------
    C : float
        Inverse regularization strength.
    l1_ratio : float
        Share of L1 in the penalty: 0 is ridge, 1 is lasso.
    class_weight : str, dict or None
        Passed to ``LogisticRegression``.
    max_iter : int
        Maximum solver iterations.
    random_state : int or None
        Seed for solvers that shuffle.
    """
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("l1_ratio must be in [0, 1]")

    kwargs = dict(
        C=C, class_weight=class_weight, max_iter=max_iter, random_state=random_state
    )
    if _PENALTY_DEPRECATED:
        solver = "saga" if 0.0 < l1_ratio < 1.0 else "liblinear"
        return LogisticRegression(l1_ratio=l1_ratio, solver=solver, **kwargs)

    if l1_ratio == 0.0:
        return LogisticRegression(penalty="l2", solver="liblinear", **kwargs)
    if l1_ratio == 1.0:
        return LogisticRegression(penalty="l1", solver="liblinear", **kwargs)
    return LogisticRegression(
        penalty="elasticnet", l1_ratio=l1_ratio, solver="saga", **kwargs
    )
