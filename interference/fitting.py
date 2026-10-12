"""Line-source inverse fitting with bounded positive permeability and storativity."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares
from .models import Reservoir, SourceWell, ObservationWell, LinearBoundary, simulate_line_source
from .validation import (K_BOUNDS_MD, STORATIVITY_BOUNDS_PER_PSI,
                         positive_number, validate_reservoir, validate_history, validate_geometry)


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
    """Fit k and phi*ct with explicit finite optimizer bounds in log space."""
    t, y = validate_history(times_hr, observed_drop_psi,
                            label="line-source pressure observations", min_points=3)
    validate_reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                       reservoir_template.B_rb_stb)
    validate_geometry(source, observation, boundaries)
    if sigma_psi is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.broadcast_to(np.asarray(sigma_psi, dtype=float), y.shape).copy()
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("All sigma_psi values must be finite and positive.")

    k0 = positive_number(k0_md, "initial permeability")
    pct0 = positive_number(phi_ct0_psi_inv, "initial phi*ct")
    for name, val, bounds in (("k0_md", k0, K_BOUNDS_MD),
                               ("phi_ct0_psi_inv", pct0, STORATIVITY_BOUNDS_PER_PSI)):
        if not (bounds[0] < val < bounds[1]):
            raise ValueError(f"{name} must lie strictly inside {bounds}; got {val}.")

    x0 = [np.log(k0), np.log(pct0)]
    lb = [np.log(K_BOUNDS_MD[0]), np.log(STORATIVITY_BOUNDS_PER_PSI[0])]
    ub = [np.log(K_BOUNDS_MD[1]), np.log(STORATIVITY_BOUNDS_PER_PSI[1])]
    if fit_offset:
        x0.append(0.0); lb.append(-1e9); ub.append(1e9)
    if fit_drift:
        x0.append(0.0); lb.append(-1e9); ub.append(1e9)

    def unpack(x):
        k, pct = float(np.exp(x[0])), float(np.exp(x[1]))
        i = 2
        offset = float(x[i]) if fit_offset else 0.0
        i += int(fit_offset)
        drift = float(x[i]) if fit_drift else 0.0
        return k, pct, offset, drift

    t0 = float(np.min(t))

    def predict(x):
        k, pct, offset, drift = unpack(x)
        res = Reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                        reservoir_template.B_rb_stb, phi_ct_psi_inv=pct)
        return simulate_line_source(t, source, observation, k, res, boundaries) + offset + drift*(t-t0)

    sol = least_squares(lambda x: (predict(x)-y)/sigma, np.asarray(x0),
                        bounds=(np.asarray(lb), np.asarray(ub)),
                        loss=robust_loss, x_scale="jac")
    pred = predict(sol.x)
    residual = pred-y
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("Nonfinite predicted pressures after fit.")
    rss = float(np.sum(residual**2))
    n, p = y.size, sol.x.size
    aic, aicc, bic = _information_criteria(rss, n, p)

    cov, stderr = None, {}
    if sol.jac is not None and n > p:
        try:
            # Approximate local errors; with robust losses this is NOT a full
            # robust covariance estimator and should not be read as a 95% CI.
            jt_j_inv = np.linalg.inv(sol.jac.T @ sol.jac)
            s2 = float(np.sum(sol.fun**2)/(n-p))
            cov_x = jt_j_inv*s2
            k, pct, _, _ = unpack(sol.x)
            jac_transform = np.eye(p)
            jac_transform[0, 0] = k
            jac_transform[1, 1] = pct
            cov = jac_transform@cov_x@jac_transform.T
            names = ["k_md", "phi_ct_psi_inv"]
            if fit_offset:
                names.append("offset_psi")
            if fit_drift:
                names.append("drift_psi_per_hr")
            stderr = {name: float(np.sqrt(max(cov[i, i], 0.0))) for i, name in enumerate(names)}
        except np.linalg.LinAlgError:
            pass
    k, pct, offset, drift = unpack(sol.x)
    return FitResult(k_md=k, phi_ct_psi_inv=pct, offset_psi=offset,
                     drift_psi_per_hr=drift, rss=rss, rmse=float(np.sqrt(rss/n)),
                     aic=float(aic), aicc=float(aicc), bic=float(bic),
                     covariance=cov, stderr=stderr, success=bool(sol.success),
                     message=str(sol.message), prediction=pred, residual=residual)
