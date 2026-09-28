"""Time-weighting kernels shared by the regression and moment-estimation layers.

Both ``src.model.regression`` (betas) and ``src.optimizer.estimators`` (factor,
residual, and sample moments) weight observations through these functions, so
"EWMA with span 252" means the same kernel at every level of the model.

Weights are ordered oldest -> newest and normalized to sum to one.
"""
from __future__ import annotations

import warnings

import numpy as np

WEIGHTINGS = ("equal", "ewma")


def resolve_decay(span: float | None = None, halflife: float | None = None) -> float:
    """Return the EWMA decay ``alpha`` from exactly one of ``span`` or ``halflife``.

    Uses the pandas conventions:

    - span N:      alpha = 2 / (N + 1)
    - halflife h:  alpha = 1 - 0.5 ** (1 / h)

    An EWMA with span N has the same mean data age, (N - 1) / 2, as an
    equal-weighted N-observation window, which makes span the natural choice
    when comparing against a trailing window.
    """
    if (span is None) == (halflife is None):
        raise ValueError("EWMA weighting requires exactly one of span or halflife")
    if span is not None:
        if span < 1:
            raise ValueError("span must be >= 1")
        return 2.0 / (span + 1.0)
    if halflife <= 0:
        raise ValueError("halflife must be > 0")
    return 1.0 - 0.5 ** (1.0 / halflife)


def validate_time_spec(
    weighting: str,
    span: float | None = None,
    halflife: float | None = None,
    stacklevel: int = 3,
) -> tuple[str, float | None, float | None]:
    """Validate a (weighting, span, halflife) spec and return it normalized.

    EWMA requires exactly one of span/halflife. A decay given with equal
    weighting warns and is dropped.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"Unknown weighting: {weighting!r}; expected one of {WEIGHTINGS}")
    if weighting == "ewma":
        resolve_decay(span=span, halflife=halflife)
        return weighting, span, halflife
    if span is not None or halflife is not None:
        warnings.warn(
            "span/halflife are ignored when weighting='equal'",
            UserWarning,
            stacklevel=stacklevel,
        )
    return weighting, None, None


def time_spec_attrs(weighting: str, span: float | None, halflife: float | None) -> dict:
    """Dataset attrs describing a time-weighting spec (None values omitted for netCDF)."""
    attrs = {"weighting": weighting}
    if span is not None:
        attrs["span"] = float(span)
    if halflife is not None:
        attrs["halflife"] = float(halflife)
    return attrs


def time_weights(
    n: int,
    weighting: str = "equal",
    span: float | None = None,
    halflife: float | None = None,
) -> np.ndarray:
    """Normalized observation weights of length ``n``, oldest first."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if weighting == "equal":
        return np.full(n, 1.0 / n)
    if weighting == "ewma":
        alpha = resolve_decay(span=span, halflife=halflife)
        ages = np.arange(n - 1, -1, -1, dtype=float)
        w = (1.0 - alpha) ** ages
        return w / w.sum()
    raise ValueError(f"Unknown weighting: {weighting!r}; expected one of {WEIGHTINGS}")


def ewma_weights(n: int, span: float | None = None, halflife: float | None = None) -> np.ndarray:
    """Normalized EWMA weights of length ``n``, oldest first."""
    return time_weights(n, "ewma", span=span, halflife=halflife)


def mean_age(weights: np.ndarray) -> float:
    """Weighted mean age (in observations) of a weight vector ordered oldest first.

    For an equal-weighted window of n this is (n - 1) / 2. For an EWMA truncated
    to a finite window it is less than the untruncated (span - 1) / 2.
    """
    w = np.asarray(weights, dtype=float)
    ages = np.arange(len(w) - 1, -1, -1, dtype=float)
    return float((w * ages).sum() / w.sum())


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())
