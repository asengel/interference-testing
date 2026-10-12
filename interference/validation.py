"""Checks shared by the CSV workflow and nonlinear fitting.

Field-unit ranges below are numerical safety bounds, not geological priors.
Values at/near a bound should not be interpreted as uniquely estimated.
"""
from __future__ import annotations

import numpy as np

# Broad finite bounds stop overflows and nonphysical optimizer excursions.
K_BOUNDS_MD = (1e-8, 1e8)
STORATIVITY_BOUNDS_PER_PSI = (1e-15, 1.0)
LEAKAGE_BOUNDS_FT = (1e-4, 1e10)
LAMBDA_BOUNDS = (1e-12, 1e4)


def positive_number(value, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.") from exc
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}.")
    return x


def finite_number(value, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; got {value!r}.") from exc
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return x


def validate_history(times, values, *, label: str, min_points: int = 1):
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or t.size != v.size:
        raise ValueError(f"{label}: time and value columns must have matching 1-D lengths.")
    if t.size < min_points:
        raise ValueError(f"{label}: at least {min_points} records are required.")
    if not np.all(np.isfinite(t)) or np.any(t < 0):
        raise ValueError(f"{label}: times must be finite and nonnegative (hours).")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{label}: times must be strictly increasing; check duplicate/out-of-order rows.")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{label}: values must all be finite; NaN/inf is not allowed.")
    return t, v


def validate_reservoir(h_ft, mu_cp, B_rb_stb, ct_psi_inv=None):
    positive_number(h_ft, "Reservoir thickness h_ft")
    positive_number(mu_cp, "Fluid viscosity mu_cp")
    positive_number(B_rb_stb, "Formation volume factor B_rb_stb")
    if ct_psi_inv is not None:
        positive_number(ct_psi_inv, "Total compressibility ct_psi_inv")


def validate_geometry(source, observation, boundary=()):
    xy = [source.x_ft, source.y_ft, observation.x_ft, observation.y_ft]
    if not np.all(np.isfinite(np.asarray(xy, float))):
        raise ValueError("Well coordinates must be finite.")
    distance = float(np.hypot(observation.x_ft-source.x_ft, observation.y_ft-source.y_ft))
    if distance <= 0:
        raise ValueError("Active and observation wells must have different coordinates.")
    for name, rw in (("Active", source.rw_ft), ("Observation", observation.rw_ft)):
        if rw is not None:
            positive_number(rw, f"{name} wellbore radius")
    for item in boundary:
        if not np.all(np.isfinite([item.x0_ft, item.y0_ft, item.angle_deg])):
            raise ValueError("Boundary coordinates and azimuth must be finite.")


def validate_fitted_parameters(fit, *, model: str, compressibility=None, boundary=False):
    if not getattr(fit, "success", False):
        raise RuntimeError(f"{model} optimizer did not converge: {getattr(fit, 'message', '')}")
    k = getattr(fit, "k_md", getattr(fit, "k_fracture_md", None))
    pct = getattr(fit, "phi_ct_psi_inv", getattr(fit, "phi_ct_total_psi_inv", None))
    for name, value, limits in (
        ("permeability", k, K_BOUNDS_MD),
        ("storativity", pct, STORATIVITY_BOUNDS_PER_PSI),
    ):
        val = positive_number(value, name)
        if not limits[0] <= val <= limits[1]:
            raise ValueError(f"Fitted {name} = {val:g} lies outside numerical bounds {limits}.")
    if compressibility is not None:
        phi = pct / compressibility
        if phi <= 0 or phi > 1:
            raise ValueError(f"Fitted porosity = (phi*ct)/ct = {phi:.4g} is outside (0,1]. ")
    for attr in ("leakage_factor_ft", "height_ft", "width_ft", "CD_active", "C_active_bbl_per_psi", "omega", "lam"):
        if hasattr(fit, attr):
            val = float(getattr(fit, attr))
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"Fitted {attr} is nonphysical: {val}.")
    if hasattr(fit, "omega") and not (0 < fit.omega <= 1):
        raise ValueError("Double-porosity omega must be in (0,1].")
    if hasattr(fit, "skin_active") and not (0 <= fit.skin_active <= 50):
        raise ValueError("Skin is outside the supported nonnegative range [0,50].")
    if not np.all(np.isfinite(fit.prediction)) or not np.isfinite(fit.rmse):
        raise ValueError("Fitted pressure predictions or RMSE are nonfinite.")
    return fit
