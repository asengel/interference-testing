from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvironmentalCorrectionResult:
    corrected_pressure: np.ndarray
    modeled_environmental_component: np.ndarray
    coefficients: dict[str, float]
    baseline_prediction: np.ndarray


def environmental_correction(
    time_hr,
    pressure,
    baseline_mask,
    barometric=None,
    tide=None,
    include_linear_drift: bool = True,
    tidal_periods_hr=(),
) -> EnvironmentalCorrectionResult:
    """Estimate and remove environmental pressure components from baseline data.

    A linear model is fitted only to samples selected by ``baseline_mask`` so the
    reservoir interference signal is not inadvertently regressed away.  Optional
    regressors are barometric pressure, a measured tide series, linear drift, and
    sinusoidal constituents with user-supplied periods.
    """
    t=np.asarray(time_hr,float); p=np.asarray(pressure,float); mask=np.asarray(baseline_mask,bool)
    if t.shape!=p.shape or mask.shape!=p.shape:
        raise ValueError("time, pressure and baseline_mask must have the same shape.")
    cols=[np.ones_like(t)]; names=["intercept"]
    if include_linear_drift:
        cols.append(t-t[mask].mean()); names.append("drift_per_hr")
    if barometric is not None:
        b=np.asarray(barometric,float)
        if b.shape!=p.shape: raise ValueError("barometric must match pressure shape.")
        cols.append(b-b[mask].mean()); names.append("barometric_coefficient")
    if tide is not None:
        td=np.asarray(tide,float)
        if td.shape!=p.shape: raise ValueError("tide must match pressure shape.")
        cols.append(td-td[mask].mean()); names.append("tide_coefficient")
    for per in tidal_periods_hr:
        if per<=0: raise ValueError("tidal periods must be positive.")
        w=2*np.pi/per
        cols.extend([np.sin(w*t),np.cos(w*t)])
        names.extend([f"sin_{per:g}h",f"cos_{per:g}h"])
    X=np.column_stack(cols)
    beta,*_=np.linalg.lstsq(X[mask],p[mask],rcond=None)
    env=X@beta
    # Preserve the baseline mean level; remove only variations around intercept.
    variations=env-beta[0]
    corrected=p-variations
    return EnvironmentalCorrectionResult(
        corrected_pressure=corrected,
        modeled_environmental_component=variations,
        coefficients={n:float(v) for n,v in zip(names,beta)},
        baseline_prediction=env,
    )
