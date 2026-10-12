"""Nonlinear fits of bounded, leaky, double-porosity and storage/skin models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from .models import (
    Reservoir, SourceWell, ObservationWell, ClosedRectangle,
    simulate_closed_rectangle, simulate_leaky_line_source, simulate_double_porosity,
)
from .validation import (
    K_BOUNDS_MD, STORATIVITY_BOUNDS_PER_PSI, LEAKAGE_BOUNDS_FT,
    LAMBDA_BOUNDS, positive_number, finite_number,
    validate_history, validate_geometry, validate_reservoir,
)


def _ic(rss: float, n: int, p: int):
    rss = max(float(rss), np.finfo(float).tiny)
    aic = n*np.log(rss/n) + 2*p
    aicc = aic + 2*p*(p+1)/(n-p-1) if n > p+1 else np.inf
    bic = n*np.log(rss/n) + p*np.log(n)
    return float(aic), float(aicc), float(bic)


def _setup(t, y, source, observation, reservoir, label):
    t, y = validate_history(t, y, label=label, min_points=5)
    validate_geometry(source, observation)
    validate_reservoir(reservoir.h_ft, reservoir.mu_cp, reservoir.B_rb_stb)
    return t, y


def _bounded_positive_start(val, limits, name):
    val = positive_number(val, name)
    if not (limits[0] < val < limits[1]):
        raise ValueError(f"{name}={val:g} must be inside numerical bounds {limits}.")
    return np.log(val)


@dataclass
class ClosedRectangleFitResult:
    k_md: float
    phi_ct_psi_inv: float
    width_ft: float
    height_ft: float
    area_ft2: float
    area_acres: float
    observation_xD: float
    observation_yD: float
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


def fit_closed_rectangle(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell, reservoir_template: Reservoir,
    aspect_ratio_width_over_height: float, source_xD: float, source_yD: float,
    k0_md: float, phi_ct0_psi_inv: float, height0_ft: float,
    rectangle_angle_deg: float = 0.0, image_layers: int | None = None,
    robust_loss: str = "linear",
) -> ClosedRectangleFitResult:
    """Fit k, phi*ct and closed-rectangle dimensions for prescribed geometry.

    The source is anchored at a known fractional rectangle position; the
    observation well offset determines its normalized location at every trial H.
    """
    aspect = positive_number(aspect_ratio_width_over_height, "rectangle aspect ratio")
    if not (0 < source_xD < 1 and 0 < source_yD < 1):
        raise ValueError("Source fractions must lie strictly between zero and one.")
    t, y = _setup(times_hr, observed_drop_psi, source, observation,
                  reservoir_template, "closed rectangle")
    ang = np.deg2rad(finite_number(rectangle_angle_deg, "rectangle angle"))
    c, s = np.cos(ang), np.sin(ang)
    dxg = observation.x_ft - source.x_ft
    dyg = observation.y_ft - source.y_ft
    dx = c*dxg + s*dyg
    dy = -s*dxg + c*dyg

    # Geometry bounds ensure both wells stay INSIDE the rectangle at every trial.
    hmin = max(
        -dx/(aspect*source_xD) if dx < 0 else dx/(aspect*(1-source_xD)),
        -dy/source_yD if dy < 0 else dy/(1-source_yD),
        0.0,
    )
    hmin = max(hmin*1.000001, 1e-5)
    hmax = max(1e9, 10*hmin)
    h0 = positive_number(height0_ft, "initial rectangle height")
    if not hmin < h0 < hmax:
        raise ValueError(f"Initial rectangle height must be between {hmin:.4g} and {hmax:.4g} ft.")

    x0 = np.array([
        _bounded_positive_start(k0_md, K_BOUNDS_MD, "initial k"),
        _bounded_positive_start(phi_ct0_psi_inv, STORATIVITY_BOUNDS_PER_PSI, "initial phi*ct"),
        np.log(h0),
    ])
    lb = np.log([K_BOUNDS_MD[0], STORATIVITY_BOUNDS_PER_PSI[0], hmin])
    ub = np.log([K_BOUNDS_MD[1], STORATIVITY_BOUNDS_PER_PSI[1], hmax])

    def unpack(x):
        k, pct, H = np.exp(x)
        W = aspect*H
        xs, ys = source_xD*W, source_yD*H
        xo, yo = xs+dx, ys+dy
        return float(k), float(pct), float(W), float(H), float(xs), float(ys), float(xo), float(yo)

    def pred(x):
        k, pct, W, H, xs, ys, xo, yo = unpack(x)
        res = Reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                        reservoir_template.B_rb_stb, phi_ct_psi_inv=pct)
        src = SourceWell(source.name, xs, ys, source.events, source.rw_ft)
        obs = ObservationWell(observation.name, xo, yo, observation.rw_ft)
        rect = ClosedRectangle(0.0, 0.0, W, H)
        return simulate_closed_rectangle(t, src, obs, k, res, rect,
                                         image_layers=image_layers)

    sol = least_squares(lambda x: pred(x)-y, x0, bounds=(lb, ub),
                        loss=robust_loss, x_scale="jac", max_nfev=1000)
    p = pred(sol.x)
    rr = p-y
    rss = float(np.sum(rr**2))
    k, pct, W, H, xs, ys, xo, yo = unpack(sol.x)
    aic, aicc, bic = _ic(rss, len(y), 3)
    return ClosedRectangleFitResult(k_md=k, phi_ct_psi_inv=pct, width_ft=W, height_ft=H,
                                    area_ft2=W*H, area_acres=W*H/43560,
                                    observation_xD=xo/W, observation_yD=yo/H,
                                    rss=rss, rmse=float(np.sqrt(rss/len(y))),
                                    aic=aic, aicc=aicc, bic=bic, prediction=p, residual=rr,
                                    success=bool(sol.success), message=str(sol.message))


@dataclass
class LeakyFitResult:
    k_md: float
    phi_ct_psi_inv: float
    leakage_factor_ft: float
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


def fit_leaky_line_source(times_hr, observed_drop_psi, source, observation,
                           reservoir_template, k0_md, phi_ct0_psi_inv,
                           leakage_factor0_ft, robust_loss="soft_l1"):
    t, y = _setup(times_hr, observed_drop_psi, source, observation,
                  reservoir_template, "leaky reservoir")
    bounds = (K_BOUNDS_MD, STORATIVITY_BOUNDS_PER_PSI, LEAKAGE_BOUNDS_FT)
    x0 = np.array([_bounded_positive_start(val, limit, label)
                   for val, limit, label in zip(
                       (k0_md, phi_ct0_psi_inv, leakage_factor0_ft), bounds,
                       ("initial k", "initial phi*ct", "initial leakage factor"))])
    lb = np.log([limit[0] for limit in bounds])
    ub = np.log([limit[1] for limit in bounds])

    def pred(x):
        k, pct, B = np.exp(x)
        res = Reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                        reservoir_template.B_rb_stb, phi_ct_psi_inv=float(pct))
        return simulate_leaky_line_source(t, source, observation, float(k), res, float(B))

    sol = least_squares(lambda x: pred(x)-y, x0, bounds=(lb, ub),
                        loss=robust_loss, x_scale="jac")
    p = pred(sol.x)
    rr = p-y
    rss = float(np.sum(rr**2))
    aic, aicc, bic = _ic(rss, len(y), 3)
    k, pct, B = np.exp(sol.x)
    return LeakyFitResult(float(k), float(pct), float(B), rss,
                          float(np.sqrt(rss/len(y))), aic, aicc, bic, p, rr,
                          bool(sol.success), str(sol.message))


@dataclass
class DoublePorosityFitResult:
    k_fracture_md: float
    phi_ct_total_psi_inv: float
    omega: float
    lam: float
    interporosity: str
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


def fit_double_porosity(times_hr, observed_drop_psi, source, observation,
                         reservoir_template, rw_ft: float, k0_md: float,
                         phi_ct_total0_psi_inv: float, omega0: float = 0.05,
                         lam0: float = 1e-4, interporosity: str = "pss",
                         robust_loss: str = "soft_l1", n_stehfest: int = 12):
    t, y = _setup(times_hr, observed_drop_psi, source, observation,
                  reservoir_template, "double porosity")
    positive_number(rw_ft, "active wellbore radius")
    if not (0 < omega0 < 1):
        raise ValueError("Initial omega must be between zero and one.")
    if interporosity not in {"pss", "transient_slab", "transient_sphere"}:
        raise ValueError("interporosity must be pss, transient_slab or transient_sphere.")
    # Bounds on logistic omega avoid division singularities at exactly zero/one.
    x0 = np.array([
        _bounded_positive_start(k0_md, K_BOUNDS_MD, "initial k"),
        _bounded_positive_start(phi_ct_total0_psi_inv, STORATIVITY_BOUNDS_PER_PSI, "initial phi*ct"),
        np.log(omega0/(1-omega0)),
        _bounded_positive_start(lam0, LAMBDA_BOUNDS, "initial lambda"),
    ])
    lb = [np.log(K_BOUNDS_MD[0]), np.log(STORATIVITY_BOUNDS_PER_PSI[0]), -16, np.log(LAMBDA_BOUNDS[0])]
    ub = [np.log(K_BOUNDS_MD[1]), np.log(STORATIVITY_BOUNDS_PER_PSI[1]), 16, np.log(LAMBDA_BOUNDS[1])]

    def unpack(x):
        return (float(np.exp(x[0])), float(np.exp(x[1])),
                float(expit(x[2])), float(np.exp(x[3])))

    def pred(x):
        k, pct, om, lam = unpack(x)
        res = Reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                        reservoir_template.B_rb_stb, phi_ct_psi_inv=pct)
        return simulate_double_porosity(t, source, observation, k, res, rw_ft,
                                        om, lam, interporosity=interporosity,
                                        n_stehfest=n_stehfest)

    sol = least_squares(lambda x: pred(x)-y, x0, bounds=(lb, ub),
                        loss=robust_loss, x_scale="jac", max_nfev=500)
    p = pred(sol.x)
    rr = p-y
    rss = float(np.sum(rr**2))
    aic, aicc, bic = _ic(rss, len(y), 4)
    k, pct, om, lam = unpack(sol.x)
    return DoublePorosityFitResult(k, pct, om, lam, interporosity, rss,
                                    float(np.sqrt(rss/len(y))), aic, aicc, bic, p, rr,
                                    bool(sol.success), str(sol.message))


@dataclass
class ActiveStorageSkinFitResult:
    k_md: float
    phi_ct_psi_inv: float
    CD_active: float
    skin_active: float
    C_active_bbl_per_psi: float
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


def fit_active_storage_skin(times_hr, observed_drop_psi, source, observation,
                            reservoir_template, k0_md: float,
                            phi_ct0_psi_inv: float, CD0: float, skin0: float,
                            robust_loss: str = "soft_l1", n_stehfest: int = 12):
    """Fit positive k, phi*ct, storage and nonnegative skin (0..50)."""
    from .models import simulate_active_storage_skin
    t, y = _setup(times_hr, observed_drop_psi, source, observation,
                  reservoir_template, "active-well storage and skin")
    if source.rw_ft is None:
        raise ValueError("Active-well rw_ft is required for storage/skin fitting.")
    positive_number(source.rw_ft, "active well radius")
    skin0 = finite_number(skin0, "initial skin")
    if not (0 <= skin0 < 50):
        raise ValueError("initial skin must be in [0,50); negative skin is not supported.")
    x0 = np.array([
        _bounded_positive_start(k0_md, K_BOUNDS_MD, "initial k"),
        _bounded_positive_start(phi_ct0_psi_inv, STORATIVITY_BOUNDS_PER_PSI, "initial phi*ct"),
        _bounded_positive_start(CD0, (1e-12, 1e14), "initial dimensionless storage"),
        skin0,
    ])
    lb = np.array([np.log(K_BOUNDS_MD[0]), np.log(STORATIVITY_BOUNDS_PER_PSI[0]),
                   np.log(1e-12), 0.0])
    ub = np.array([np.log(K_BOUNDS_MD[1]), np.log(STORATIVITY_BOUNDS_PER_PSI[1]),
                   np.log(1e14), 50.0])

    def unpack(x):
        return float(np.exp(x[0])), float(np.exp(x[1])), float(np.exp(x[2])), float(x[3])

    def pred(x):
        k, pct, CD, skin = unpack(x)
        res = Reservoir(reservoir_template.h_ft, reservoir_template.mu_cp,
                        reservoir_template.B_rb_stb, phi_ct_psi_inv=pct)
        return simulate_active_storage_skin(t, source, observation, k, res,
                                            CD, skin, n_stehfest=n_stehfest)

    sol = least_squares(lambda x: pred(x)-y, x0, bounds=(lb, ub),
                        loss=robust_loss, x_scale="jac", max_nfev=800)
    p = pred(sol.x)
    rr = p-y
    rss = float(np.sum(rr**2))
    aic, aicc, bic = _ic(rss, len(y), 4)
    k, pct, CD, skin = unpack(sol.x)
    C = CD*(2*np.pi*pct*reservoir_template.h_ft*source.rw_ft**2)/5.615
    return ActiveStorageSkinFitResult(k, pct, CD, skin, float(C), rss,
        float(np.sqrt(rss/len(y))), aic, aicc, bic, p, rr,
        bool(sol.success), str(sol.message))
