from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares

from .models import (
    Reservoir, SourceWell, ObservationWell, ClosedRectangle,
    simulate_closed_rectangle, simulate_leaky_line_source, simulate_double_porosity,
)


def _ic(rss: float, n: int, p: int):
    rss = max(float(rss), np.finfo(float).tiny)
    aic = n*np.log(rss/n) + 2*p
    aicc = aic + 2*p*(p+1)/(n-p-1) if n > p+1 else np.inf
    bic = n*np.log(rss/n) + p*np.log(n)
    return float(aic), float(aicc), float(bic)


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
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    aspect_ratio_width_over_height: float,
    source_xD: float,
    source_yD: float,
    k0_md: float,
    phi_ct0_psi_inv: float,
    height0_ft: float,
    rectangle_angle_deg: float = 0.0,
    image_layers: int | None = None,
    robust_loss: str = "linear",
) -> ClosedRectangleFitResult:
    """Fit k, phi*ct and closed-rectangle size from one interference pair.

    The rectangle aspect ratio, source normalized position, and rectangle
    orientation are supplied from geology/structure.  The source well anchors
    the rectangle location.  The observation-well physical offset then changes
    normalized position as rectangle size changes, which makes reservoir size
    estimable when the bounded response is visible.
    """
    if aspect_ratio_width_over_height <= 0:
        raise ValueError("aspect ratio must be positive.")
    if not (0 < source_xD < 1 and 0 < source_yD < 1):
        raise ValueError("source normalized coordinates must lie strictly inside rectangle.")
    t = np.asarray(times_hr, float); y = np.asarray(observed_drop_psi, float)
    if t.shape != y.shape:
        raise ValueError("times and observations must have the same shape.")

    ang = np.deg2rad(rectangle_angle_deg)
    c, s = np.cos(ang), np.sin(ang)
    dxg = observation.x_ft - source.x_ft
    dyg = observation.y_ft - source.y_ft
    # Global -> rectangle-local offset.
    dx = c*dxg + s*dyg
    dy = -s*dxg + c*dyg

    x0 = np.log([k0_md, phi_ct0_psi_inv, height0_ft])

    def unpack(x):
        k, pct, H = np.exp(x)
        W = aspect_ratio_width_over_height * H
        xs, ys = source_xD*W, source_yD*H
        xo, yo = xs + dx, ys + dy
        return float(k), float(pct), float(W), float(H), float(xs), float(ys), float(xo), float(yo)

    def pred(x):
        k,pct,W,H,xs,ys,xo,yo = unpack(x)
        if not (0 < xo < W and 0 < yo < H):
            return np.full_like(y, 1e8)
        res = Reservoir(
            h_ft=reservoir_template.h_ft,
            mu_cp=reservoir_template.mu_cp,
            B_rb_stb=reservoir_template.B_rb_stb,
            phi_ct_psi_inv=pct,
        )
        src = SourceWell(source.name, xs, ys, source.events, source.rw_ft)
        obs = ObservationWell(observation.name, xo, yo, observation.rw_ft)
        rect = ClosedRectangle(0.0, 0.0, W, H)
        return simulate_closed_rectangle(t, src, obs, k, res, rect, image_layers=image_layers)

    sol = least_squares(lambda x: pred(x)-y, x0, loss=robust_loss, x_scale="jac", max_nfev=1000)
    p = pred(sol.x); rr = p-y; rss=float(np.sum(rr**2))
    k,pct,W,H,xs,ys,xo,yo = unpack(sol.x)
    aic,aicc,bic=_ic(rss,len(y),3)
    area=W*H
    return ClosedRectangleFitResult(
        k_md=k, phi_ct_psi_inv=pct, width_ft=W, height_ft=H,
        area_ft2=area, area_acres=area/43560.0,
        observation_xD=xo/W, observation_yD=yo/H,
        rss=rss, rmse=float(np.sqrt(rss/len(y))), aic=aic, aicc=aicc, bic=bic,
        prediction=p, residual=rr, success=bool(sol.success), message=str(sol.message)
    )


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


def fit_leaky_line_source(
    times_hr, observed_drop_psi, source, observation, reservoir_template,
    k0_md, phi_ct0_psi_inv, leakage_factor0_ft,
    robust_loss: str = "soft_l1",
):
    t=np.asarray(times_hr,float); y=np.asarray(observed_drop_psi,float)
    x0=np.log([k0_md,phi_ct0_psi_inv,leakage_factor0_ft])
    def pred(x):
        k,pct,B=np.exp(x)
        res=Reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,
                      reservoir_template.B_rb_stb,phi_ct_psi_inv=float(pct))
        return simulate_leaky_line_source(t,source,observation,float(k),res,float(B))
    sol=least_squares(lambda x:pred(x)-y,x0,loss=robust_loss,x_scale="jac")
    p=pred(sol.x); rr=p-y; rss=float(np.sum(rr**2)); aic,aicc,bic=_ic(rss,len(y),3)
    k,pct,B=np.exp(sol.x)
    return LeakyFitResult(float(k),float(pct),float(B),rss,float(np.sqrt(rss/len(y))),
                          aic,aicc,bic,p,rr,bool(sol.success),str(sol.message))


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


def fit_double_porosity(
    times_hr, observed_drop_psi, source, observation, reservoir_template,
    rw_ft: float, k0_md: float, phi_ct_total0_psi_inv: float,
    omega0: float = 0.05, lam0: float = 1e-4,
    interporosity: str = "pss", robust_loss: str = "soft_l1",
    n_stehfest: int = 12,
):
    """Fit k_f, total phi*ct, omega and lambda for a double-porosity model."""
    t=np.asarray(times_hr,float); y=np.asarray(observed_drop_psi,float)
    if not (0 < omega0 < 1):
        raise ValueError("omega0 must be between zero and one.")
    # logit omega, logs for positive parameters
    z0=np.log(omega0/(1-omega0))
    x0=np.array([np.log(k0_md),np.log(phi_ct_total0_psi_inv),z0,np.log(lam0)])
    def unpack(x):
        k=np.exp(x[0]); pct=np.exp(x[1]); om=1/(1+np.exp(-x[2])); lam=np.exp(x[3])
        return float(k),float(pct),float(om),float(lam)
    def pred(x):
        k,pct,om,lam=unpack(x)
        res=Reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,
                      reservoir_template.B_rb_stb,phi_ct_psi_inv=pct)
        return simulate_double_porosity(t,source,observation,k,res,rw_ft,om,lam,
                                        interporosity=interporosity,n_stehfest=n_stehfest)
    sol=least_squares(lambda x:pred(x)-y,x0,loss=robust_loss,x_scale="jac",max_nfev=500)
    p=pred(sol.x); rr=p-y; rss=float(np.sum(rr**2)); aic,aicc,bic=_ic(rss,len(y),4)
    k,pct,om,lam=unpack(sol.x)
    return DoublePorosityFitResult(k,pct,om,lam,interporosity,rss,float(np.sqrt(rss/len(y))),
                                   aic,aicc,bic,p,rr,bool(sol.success),str(sol.message))


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


def fit_active_storage_skin(
    times_hr, observed_drop_psi, source, observation, reservoir_template,
    k0_md: float, phi_ct0_psi_inv: float, CD0: float, skin0: float,
    robust_loss: str = "soft_l1", n_stehfest: int = 12,
):
    """Fit k, phi*ct, active-well dimensionless storage and nonnegative skin.

    The active well radius is a known geometric well input in ``source.rw_ft``.
    Physical wellbore storage C is reported from the fitted dimensionless CD.
    """
    from .models import simulate_active_storage_skin

    if source.rw_ft is None or source.rw_ft <= 0:
        raise ValueError("Active-well rw_ft is required for storage/skin fitting.")
    if skin0 < 0:
        raise ValueError("skin0 must be nonnegative; negative-skin Stehfest inversion is blocked.")

    t = np.asarray(times_hr, float)
    y = np.asarray(observed_drop_psi, float)
    if t.shape != y.shape:
        raise ValueError("times and observations must have the same shape.")

    x0 = np.array([np.log(k0_md), np.log(phi_ct0_psi_inv), np.log(max(CD0, 1e-12)), skin0], float)
    lb = np.array([np.log(1e-8), np.log(1e-15), np.log(1e-12), 0.0])
    ub = np.array([np.log(1e8), np.log(1.0), np.log(1e14), 50.0])

    def unpack(x):
        return float(np.exp(x[0])), float(np.exp(x[1])), float(np.exp(x[2])), float(x[3])

    def pred(x):
        k, pct, CD, skin = unpack(x)
        res = Reservoir(
            reservoir_template.h_ft, reservoir_template.mu_cp,
            reservoir_template.B_rb_stb, phi_ct_psi_inv=pct,
        )
        return simulate_active_storage_skin(
            t, source, observation, k, res, CD, skin, n_stehfest=n_stehfest
        )

    sol = least_squares(
        lambda x: pred(x) - y, x0, bounds=(lb, ub), loss=robust_loss,
        x_scale="jac", max_nfev=800,
    )
    p = pred(sol.x)
    rr = p - y
    rss = float(np.sum(rr**2))
    aic, aicc, bic = _ic(rss, len(y), 4)
    k, pct, CD, skin = unpack(sol.x)
    # van Everdingen-Hurst field-unit definition:
    # CD = 5.615 C / (2*pi*phi*ct*h*rw^2)
    C = CD * (2.0*np.pi*pct*reservoir_template.h_ft*source.rw_ft**2) / 5.615
    return ActiveStorageSkinFitResult(
        k, pct, CD, skin, float(C), rss, float(np.sqrt(rss/len(y))),
        aic, aicc, bic, p, rr, bool(sol.success), str(sol.message)
    )
