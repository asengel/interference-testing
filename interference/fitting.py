from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares

from .models import Reservoir, SourceWell, ObservationWell, LinearBoundary, simulate_line_source


@dataclass
class FitResult:
    k_md: float
    phi_ct_psi_inv: float
    offset_psi: float
    drift_psi_per_hr: float
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    covariance: np.ndarray | None
    stderr: dict[str, float]
    success: bool
    message: str
    prediction: np.ndarray
    residual: np.ndarray


def _information_criteria(rss: float, n: int, p: int):
    rss = max(float(rss), np.finfo(float).tiny)
    aic = n * np.log(rss / n) + 2 * p
    aicc = aic + (2 * p * (p + 1) / (n - p - 1)) if n > p + 1 else np.inf
    bic = n * np.log(rss / n) + p * np.log(n)
    return aic, aicc, bic


def fit_k_phi_ct(
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    k0_md: float,
    phi_ct0_psi_inv: float,
    boundaries: Sequence[LinearBoundary] = (),
    sigma_psi: Sequence[float] | float | None = None,
    fit_offset: bool = False,
    fit_drift: bool = False,
    robust_loss: str = "soft_l1",
) -> FitResult:
    """Fit permeability and phi*ct by nonlinear least squares in log-parameter space."""
    t = np.asarray(times_hr, dtype=float)
    y = np.asarray(observed_drop_psi, dtype=float)
    if t.shape != y.shape:
        raise ValueError("times and observed data must have the same shape.")
    if sigma_psi is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.broadcast_to(np.asarray(sigma_psi, dtype=float), y.shape).copy()
        if np.any(sigma <= 0):
            raise ValueError("All sigma values must be positive.")

    x0 = [np.log(k0_md), np.log(phi_ct0_psi_inv)]
    if fit_offset:
        x0.append(0.0)
    if fit_drift:
        x0.append(0.0)

    def unpack(x):
        k = float(np.exp(x[0]))
        pct = float(np.exp(x[1]))
        j = 2
        offset = float(x[j]) if fit_offset else 0.0
        j += int(fit_offset)
        drift = float(x[j]) if fit_drift else 0.0
        return k, pct, offset, drift

    t0 = float(np.min(t))

    def predict(x):
        k, pct, offset, drift = unpack(x)
        res = Reservoir(
            h_ft=reservoir_template.h_ft,
            mu_cp=reservoir_template.mu_cp,
            B_rb_stb=reservoir_template.B_rb_stb,
            phi_ct_psi_inv=pct,
        )
        base = simulate_line_source(t, source, observation, k, res, boundaries)
        return base + offset + drift * (t - t0)

    def residuals(x):
        return (predict(x) - y) / sigma

    sol = least_squares(residuals, x0, loss=robust_loss, x_scale="jac")
    pred = predict(sol.x)
    raw_resid = pred - y
    rss = float(np.sum(raw_resid**2))
    n = y.size
    p = sol.x.size
    aic, aicc, bic = _information_criteria(rss, n, p)

    cov = None
    stderr = {}
    if sol.jac is not None and n > p:
        try:
            jt_j_inv = np.linalg.inv(sol.jac.T @ sol.jac)
            s2 = float(np.sum(sol.fun**2) / (n - p))
            cov_x = jt_j_inv * s2
            k, pct, offset, drift = unpack(sol.x)
            # delta method from log parameters to physical parameters
            Jtr = np.eye(p)
            Jtr[0, 0] = k
            Jtr[1, 1] = pct
            cov = Jtr @ cov_x @ Jtr.T
            names = ["k_md", "phi_ct_psi_inv"]
            if fit_offset:
                names.append("offset_psi")
            if fit_drift:
                names.append("drift_psi_per_hr")
            stderr = {name: float(np.sqrt(max(cov[i, i], 0.0))) for i, name in enumerate(names)}
        except np.linalg.LinAlgError:
            pass

    k, pct, offset, drift = unpack(sol.x)
    return FitResult(
        k_md=k,
        phi_ct_psi_inv=pct,
        offset_psi=offset,
        drift_psi_per_hr=drift,
        rss=rss,
        rmse=float(np.sqrt(rss / n)),
        aic=float(aic),
        aicc=float(aicc),
        bic=float(bic),
        covariance=cov,
        stderr=stderr,
        success=bool(sol.success),
        message=str(sol.message),
        prediction=pred,
        residual=raw_resid,
    )
