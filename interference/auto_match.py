"""Optional inverse fitting of straight boundaries and homogeneous anisotropy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares
from .models import (Reservoir, SourceWell, ObservationWell, LinearBoundary,
                     simulate_line_source, simulate_anisotropic)
from .fitting import FitResult, fit_k_phi_ct
from .validation import (K_BOUNDS_MD, STORATIVITY_BOUNDS_PER_PSI,
                         finite_number, positive_number, validate_history,
                         validate_geometry, validate_reservoir)


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
    aic = n*np.log(rss/n)+2*p
    aicc = aic + 2*p*(p+1)/(n-p-1) if n>p+1 else np.inf
    bic = n*np.log(rss/n)+p*np.log(n)
    return float(aic),float(aicc),float(bic)


def fit_single_linear_boundary(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell,
    reservoir_template: Reservoir, kind: str, normal_angle_deg: float,
    k0_md: float, phi_ct0_psi_inv: float, boundary_distance0_ft: float,
    robust_loss: str = "soft_l1",
) -> BoundaryFitResult:
    """Fit k, phi*ct and distance for a specified straight-boundary azimuth.

    The orientation is fixed because fitting it together with distance from
    only one observation well is generally non-identifiable.
    """
    if kind not in {"no_flow", "constant_pressure"}:
        raise ValueError("kind must be no_flow or constant_pressure.")
    t,y=validate_history(times_hr,observed_drop_psi,label="boundary fit",min_points=5)
    validate_reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,reservoir_template.B_rb_stb)
    validate_geometry(source,observation)
    theta=finite_number(normal_angle_deg,"boundary normal azimuth")
    a=np.deg2rad(theta)
    nx,ny=np.cos(a),np.sin(a)
    projected=(observation.x_ft-source.x_ft)*nx+(observation.y_ft-source.y_ft)*ny
    # The trial wall must keep both wells on the same reservoir side.
    dmin=max(1e-5,projected+max(abs(projected)*1e-5,1e-5))
    dmax=max(1e10,dmin*10)
    d0=positive_number(boundary_distance0_ft,"initial boundary distance")
    if not (dmin<d0<dmax):
        raise ValueError(f"initial boundary distance must lie inside ({dmin:g}, {dmax:g}) ft.")
    k0=positive_number(k0_md,"initial k")
    pct0=positive_number(phi_ct0_psi_inv,"initial phi*ct")
    if not (K_BOUNDS_MD[0]<k0<K_BOUNDS_MD[1] and
            STORATIVITY_BOUNDS_PER_PSI[0]<pct0<STORATIVITY_BOUNDS_PER_PSI[1]):
        raise ValueError("Initial k or phi*ct lies outside numerical bounds.")
    x0=np.log([k0,pct0,d0])
    lb=np.log([K_BOUNDS_MD[0],STORATIVITY_BOUNDS_PER_PSI[0],dmin])
    ub=np.log([K_BOUNDS_MD[1],STORATIVITY_BOUNDS_PER_PSI[1],dmax])

    def pred(x):
        k,pct,d=np.exp(x)
        res=Reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,
                      reservoir_template.B_rb_stb,phi_ct_psi_inv=float(pct))
        b=LinearBoundary(source.x_ft+d*nx,source.y_ft+d*ny,theta,kind)
        return simulate_line_source(t,source,observation,float(k),res,[b])

    sol=least_squares(lambda x:pred(x)-y,x0,bounds=(lb,ub),
                      loss=robust_loss,x_scale="jac")
    p=pred(sol.x)
    rr=p-y
    rss=float(np.sum(rr**2))
    aic,aicc,bic=_ic(rss,len(y),3)
    k,pct,d=np.exp(sol.x)
    return BoundaryFitResult(kind,float(k),float(pct),float(d),theta,
        rss,float(np.sqrt(rss/len(y))),aic,aicc,bic,p,rr,bool(sol.success),str(sol.message))


def compare_basic_candidates(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell,
    reservoir_template: Reservoir, k0_md: float, phi_ct0_psi_inv: float,
    boundary_normal_angle_deg: float | None = None,
    boundary_distance0_ft: float | None = None,
) -> list[CandidateResult]:
    base=fit_k_phi_ct(times_hr,observed_drop_psi,source,observation,
                      reservoir_template,k0_md,phi_ct0_psi_inv,robust_loss="soft_l1")
    results=[CandidateResult("infinite_line_source",base.aicc,base.bic,base.rmse,
        {"k_md":base.k_md,"phi_ct_psi_inv":base.phi_ct_psi_inv},base)]
    if boundary_normal_angle_deg is not None and boundary_distance0_ft is not None:
        for kind in ("no_flow","constant_pressure"):
            fit=fit_single_linear_boundary(times_hr,observed_drop_psi,source,observation,
                reservoir_template,kind,boundary_normal_angle_deg,k0_md,
                phi_ct0_psi_inv,boundary_distance0_ft)
            results.append(CandidateResult(f"linear_{kind}",fit.aicc,fit.bic,fit.rmse,
                {"k_md":fit.k_md,"phi_ct_psi_inv":fit.phi_ct_psi_inv,
                 "boundary_distance_ft":fit.boundary_distance_ft,
                 "normal_angle_deg":fit.normal_angle_deg},fit))
    return sorted(results,key=lambda z:z.aicc)


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
    source: SourceWell, observations: Sequence[ObservationWell],
    reservoir_template: Reservoir, kgeom0_md: float, anisotropy_ratio0: float,
    theta0_deg: float, phi_ct0_psi_inv: float,
) -> AnisotropyFitResult:
    if len(observations)<2:
        raise ValueError("At least two observation wells are required for anisotropy fitting.")
    if not (len(times_by_observation)==len(drops_by_observation)==len(observations)):
        raise ValueError("Supply one time and pressure array per observation well.")
    times=[]; ys=[]
    for t,y,obs in zip(times_by_observation,drops_by_observation,observations):
        t,y=validate_history(t,y,label=f"anisotropy {obs.name}",min_points=4)
        validate_geometry(source,obs)
        times.append(t); ys.append(y)
    validate_reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,
                       reservoir_template.B_rb_stb)
    kg=positive_number(kgeom0_md,"initial geometric mean k")
    ratio=positive_number(anisotropy_ratio0,"initial anisotropy ratio")
    pct=positive_number(phi_ct0_psi_inv,"initial phi*ct")
    theta=finite_number(theta0_deg,"initial anisotropy azimuth")
    if not (K_BOUNDS_MD[0]<kg<K_BOUNDS_MD[1] and 1<=ratio<1e4 and
            -180<=theta<=180 and STORATIVITY_BOUNDS_PER_PSI[0]<pct<STORATIVITY_BOUNDS_PER_PSI[1]):
        raise ValueError("Initial anisotropy fit values are outside allowed bounds.")
    x0=np.array([np.log(kg),np.log(ratio),theta,np.log(pct)])
    lb=np.array([np.log(K_BOUNDS_MD[0]),0.,-180.,np.log(STORATIVITY_BOUNDS_PER_PSI[0])])
    ub=np.array([np.log(K_BOUNDS_MD[1]),np.log(1e4),180.,np.log(STORATIVITY_BOUNDS_PER_PSI[1])])

    def unpack(x):
        k=np.exp(x[0]); ratio=np.exp(x[1]); theta=x[2]; pct=np.exp(x[3]); root=np.sqrt(ratio)
        return k*root,k/root,theta,pct

    def residual(x):
        kx,ky,theta,pct=unpack(x)
        res=Reservoir(reservoir_template.h_ft,reservoir_template.mu_cp,
                      reservoir_template.B_rb_stb,phi_ct_psi_inv=float(pct))
        return np.concatenate([simulate_anisotropic(t,source,obs,kx,ky,theta,res)-y
                               for t,y,obs in zip(times,ys,observations)])

    sol=least_squares(residual,x0,bounds=(lb,ub),loss="soft_l1",x_scale="jac")
    rr=residual(sol.x)
    kx,ky,theta,pct=unpack(sol.x)
    return AnisotropyFitResult(float(kx),float(ky),float(theta),float(pct),
        float(np.sum(rr**2)),float(np.sqrt(np.mean(rr**2))),bool(sol.success),str(sol.message))
