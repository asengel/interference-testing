from __future__ import annotations
import numpy as np


def bourdet_derivative(time: np.ndarray, pressure: np.ndarray, L: float = 0.2):
    """Bourdet-style derivative dP/dln(t) using logarithmic neighbor spacing.

    Returns (time_mid, derivative); endpoints or nonpositive times are excluded.
    L is a target log10 smoothing distance used to choose left/right neighbors.
    """
    t = np.asarray(time, dtype=float)
    p = np.asarray(pressure, dtype=float)
    mask = np.isfinite(t) & np.isfinite(p) & (t > 0)
    t, p = t[mask], p[mask]
    order = np.argsort(t)
    t, p = t[order], p[order]
    x = np.log(t)
    target = L * np.log(10.0)
    deriv = np.full_like(p, np.nan)
    for i in range(1, len(t) - 1):
        left = i - 1
        while left > 0 and x[i] - x[left] < target:
            left -= 1
        right = i + 1
        while right < len(t) - 1 and x[right] - x[i] < target:
            right += 1
        dx1 = x[i] - x[left]
        dx2 = x[right] - x[i]
        if dx1 <= 0 or dx2 <= 0:
            continue
        m1 = (p[i] - p[left]) / dx1
        m2 = (p[right] - p[i]) / dx2
        deriv[i] = (m1 * dx2 + m2 * dx1) / (dx1 + dx2)
    good = np.isfinite(deriv)
    return t[good], deriv[good]


def residual_stats(residual: np.ndarray) -> dict[str, float]:
    r = np.asarray(residual, dtype=float)
    return {
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(r**2))),
        "max_abs": float(np.max(np.abs(r))) if r.size else 0.0,
    }
