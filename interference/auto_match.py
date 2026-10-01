from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares

from .models import (
    Reservoir, SourceWell, ObservationWell, LinearBoundary,
    simulate_line_source, simulate_anisotropic,
)
from .fitting import FitResult, fit_k_phi_ct


@dataclass
class BoundaryFitResult:
    kind: str
    k_md: float
    phi_ct_psi_inv: float
    boundary_distance_ft: float
    normal_angle_deg: float
    rss: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    prediction: np.ndarray
    residual: np.ndarray
    success: bool
    message: str


@dataclass
class CandidateResult:
    model: str
    aicc: float
    bic: float
    rmse: float
    parameters: dict[str, float]
    result: object


def _ic(rss: float, n: int, p: int):
    rss = max(float(rss), np.finfo(float).tiny)
    aic = n * np.log(rss / n) + 2*p
    aicc = aic + 2*p*(p+1)/(n-p-1) if n > p+1 else np.inf
    bic = n * np.log(rss/n) + p*np.log(n)
    return float(aic), float(aicc), float(bic)


def fit_single_linear_boundary(
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    kind: str,
    normal_angle_deg: float,
    k0_md: float,
    phi_ct0_psi_inv: float,
    boundary_distance0_ft: float,
    robust_loss: str = "soft_l1",
) -> BoundaryFitResult:
    """Fit k, phi*ct and source-to-boundary distance for a specified boundary orientation.

    The boundary normal points from the source toward the boundary. With only one
    source-observation pair, fitting both boundary orientation and distance is generally
    nonunique, so orientation is intentionally supplied rather than silently overfit.
    """
    if kind not in {"no_flow", "constant_pressure"}:
        raise ValueError("kind must be 'no_flow' or 'constant_pressure'.")
    t = np.asarray(times_hr, float)
    y = np.asarray(observed_drop_psi, float)
    if t.shape != y.shape:
        raise ValueError("times and observations must have the same shape.")

    a = np.deg2rad(normal_angle_deg)
    nx, ny = np.cos(a), np.sin(a)
    x0 = np.log([k0_md, phi_ct0_psi_inv, boundary_distance0_ft])

    def pred(x):
        k, pct, d = np.exp(x)
        res = Reservoir(
            h_ft=reservoir_template.h_ft,
            mu_cp=reservoir_template.mu_cp,
            B_rb_stb=reservoir_template.B_rb_stb,
            phi_ct_psi_inv=float(pct),
        )
        b = LinearBoundary(
            source.x_ft + d*nx,
            source.y_ft + d*ny,
            normal_angle_deg,
            kind,
        )
        return simulate_line_source(t, source, observation, float(k), res, [b])

    sol = least_squares(lambda x: pred(x)-y, x0, loss=robust_loss, x_scale="jac")
    prediction = pred(sol.x)
    residual = prediction-y
    rss = float(np.sum(residual**2))
    aic, aicc, bic = _ic(rss, len(y), 3)
    k,pct,d = np.exp(sol.x)
    return BoundaryFitResult(
        kind=kind, k_md=float(k), phi_ct_psi_inv=float(pct),
        boundary_distance_ft=float(d), normal_angle_deg=float(normal_angle_deg),
        rss=rss, rmse=float(np.sqrt(rss/len(y))), aic=aic, aicc=aicc, bic=bic,
        prediction=prediction, residual=residual, success=bool(sol.success), message=str(sol.message)
    )


def compare_basic_candidates(
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    k0_md: float,
    phi_ct0_psi_inv: float,
    boundary_normal_angle_deg: float | None = None,
    boundary_distance0_ft: float | None = None,
) -> list[CandidateResult]:
    """Fit and compare infinite, no-flow and constant-pressure candidates.

    Boundary candidates are only run when a physically motivated orientation and initial
    distance are provided. Results are returned sorted by AICc, but the caller should also
    inspect parameter plausibility and residual diagnostics before accepting a model.
    """
    base = fit_k_phi_ct(
        times_hr, observed_drop_psi, source, observation, reservoir_template,
        k0_md, phi_ct0_psi_inv, robust_loss="soft_l1"
    )
    out = [CandidateResult(
        model="infinite_line_source", aicc=base.aicc, bic=base.bic, rmse=base.rmse,
        parameters={"k_md":base.k_md,"phi_ct_psi_inv":base.phi_ct_psi_inv}, result=base
    )]
    if boundary_normal_angle_deg is not None and boundary_distance0_ft is not None:
        for kind in ("no_flow", "constant_pressure"):
            r = fit_single_linear_boundary(
                times_hr, observed_drop_psi, source, observation, reservoir_template,
                kind, boundary_normal_angle_deg, k0_md, phi_ct0_psi_inv,
                boundary_distance0_ft
            )
            out.append(CandidateResult(
                model=f"linear_{kind}", aicc=r.aicc, bic=r.bic, rmse=r.rmse,
                parameters={
                    "k_md":r.k_md, "phi_ct_psi_inv":r.phi_ct_psi_inv,
                    "boundary_distance_ft":r.boundary_distance_ft,
                    "normal_angle_deg":r.normal_angle_deg,
                }, result=r
            ))
    return sorted(out, key=lambda z: z.aicc)


@dataclass
class AnisotropyFitResult:
    kx_md: float
    ky_md: float
    theta_deg: float
    phi_ct_psi_inv: float
    rss: float
    rmse: float
    success: bool
    message: str


def fit_joint_anisotropy(
    times_by_observation: Sequence[Sequence[float]],
    drops_by_observation: Sequence[Sequence[float]],
    source: SourceWell,
    observations: Sequence[ObservationWell],
    reservoir_template: Reservoir,
    kgeom0_md: float,
    anisotropy_ratio0: float,
    theta0_deg: float,
    phi_ct0_psi_inv: float,
) -> AnisotropyFitResult:
    """Jointly fit homogeneous permeability anisotropy to >=2 observation wells."""
    if len(observations) < 2:
        raise ValueError("At least two observation wells are required for anisotropy fitting.")
    if not (len(times_by_observation)==len(drops_by_observation)==len(observations)):
        raise ValueError("Provide one time/data array per observation well.")

    times = [np.asarray(v,float) for v in times_by_observation]
    ys = [np.asarray(v,float) for v in drops_by_observation]
    for t,y in zip(times,ys):
        if t.shape != y.shape:
            raise ValueError("Each time array must match its data array.")

    x0 = np.array([np.log(kgeom0_md), np.log(max(anisotropy_ratio0,1.0)), theta0_deg, np.log(phi_ct0_psi_inv)])
    lb = np.array([np.log(1e-8), 0.0, -180.0, np.log(1e-15)])
    ub = np.array([np.log(1e8), np.log(1e4), 180.0, np.log(1.0)])

    def unpack(x):
        kg = np.exp(x[0]); ratio=np.exp(x[1]); theta=x[2]; pct=np.exp(x[3])
        root=np.sqrt(ratio)
        return kg*root, kg/root, theta, pct

    def residual(x):
        kx,ky,theta,pct=unpack(x)
        res=Reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,reservoir_template.B_rb_stb,phi_ct_psi_inv=pct)
        rr=[]
        for t,y,obs in zip(times,ys,observations):
            rr.append(simulate_anisotropic(t,source,obs,kx,ky,theta,res)-y)
        return np.concatenate(rr)

    sol=least_squares(residual,x0,bounds=(lb,ub),loss="soft_l1",x_scale="jac")
    rr=residual(sol.x)
    kx,ky,theta,pct=unpack(sol.x)
    return AnisotropyFitResult(
        kx_md=float(kx), ky_md=float(ky), theta_deg=float(theta), phi_ct_psi_inv=float(pct),
        rss=float(np.sum(rr**2)), rmse=float(np.sqrt(np.mean(rr**2))), success=bool(sol.success), message=str(sol.message)
    )
