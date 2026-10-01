from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .models import Reservoir, SourceWell, ObservationWell
from .fitting import fit_k_phi_ct, FitResult
from .advanced_fitting import fit_closed_rectangle, ClosedRectangleFitResult


# Internal numerical start points only. These are NOT engineering inputs and are
# intentionally hidden from the CSV workflow. They are used to make the nonlinear
# inverse problem robust over petroleum/geothermal permeability and storativity ranges.
_K_STARTS_MD = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
_PCT_STARTS = np.array([1e-9, 1e-7, 1e-6, 1e-5])


def _candidate_score(result) -> float:
    if not getattr(result, "success", False):
        return np.inf
    rmse = float(getattr(result, "rmse", np.inf))
    if not np.isfinite(rmse):
        return np.inf
    return rmse


def fit_line_source_auto(
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    robust_loss: str = "soft_l1",
) -> FitResult:
    """Automatically fit k and phi*ct without user-supplied initial guesses.

    Multiple internal starts are tried in log-parameter space. The best converged
    fit is returned. The start values are numerical solver aids, not model inputs.
    """
    best: FitResult | None = None

    # A coarse cross-grid is deliberately robust. Keep only a limited set by
    # pairing permeability and storativity scales around a range of diffusivities.
    starts: list[tuple[float, float]] = []
    for k in _K_STARTS_MD:
        for pct in _PCT_STARTS:
            starts.append((float(k), float(pct)))

    for k0, pct0 in starts:
        try:
            fit = fit_k_phi_ct(
                times_hr,
                observed_drop_psi,
                source,
                observation,
                reservoir_template,
                k0_md=k0,
                phi_ct0_psi_inv=pct0,
                robust_loss=robust_loss,
            )
        except Exception:
            continue
        if best is None or _candidate_score(fit) < _candidate_score(best):
            best = fit

    if best is None:
        raise RuntimeError("Automatic line-source matching failed for all internal start points.")
    return best


def _minimum_rectangle_height(
    source: SourceWell,
    observation: ObservationWell,
    aspect_ratio_width_over_height: float,
    source_xD: float,
    source_yD: float,
    rectangle_angle_deg: float,
) -> float:
    """Minimum H that keeps source and observation inside the rectangle."""
    if aspect_ratio_width_over_height <= 0:
        raise ValueError("Rectangle aspect ratio must be positive.")
    if not (0.0 < source_xD < 1.0 and 0.0 < source_yD < 1.0):
        raise ValueError("source_xD and source_yD must lie strictly between 0 and 1.")

    a = np.deg2rad(rectangle_angle_deg)
    c, s = np.cos(a), np.sin(a)
    dxg = observation.x_ft - source.x_ft
    dyg = observation.y_ft - source.y_ft
    dx = c * dxg + s * dyg
    dy = -s * dxg + c * dyg
    A = aspect_ratio_width_over_height

    hmin = 0.0
    if dx < 0:
        hmin = max(hmin, -dx / (source_xD * A))
    elif dx > 0:
        hmin = max(hmin, dx / ((1.0 - source_xD) * A))
    if dy < 0:
        hmin = max(hmin, -dy / source_yD)
    elif dy > 0:
        hmin = max(hmin, dy / (1.0 - source_yD))

    # Always keep a finite margin from a boundary.
    interwell = float(np.hypot(dxg, dyg))
    return max(1.05 * hmin, 0.5 * interwell, 1.0)


def fit_closed_rectangle_auto(
    times_hr: Sequence[float],
    observed_drop_psi: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    reservoir_template: Reservoir,
    aspect_ratio_width_over_height: float,
    source_xD: float,
    source_yD: float,
    rectangle_angle_deg: float = 0.0,
    robust_loss: str = "soft_l1",
) -> ClosedRectangleFitResult:
    """Fit k, phi*ct and rectangle size without user-supplied start parameters.

    A line-source fit supplies a data-driven hydraulic starting scale, then several
    internally generated reservoir-size starts are tried. Only geological quantities
    that define the selected model (aspect ratio, source position, orientation) are
    supplied by the user.
    """
    base = fit_line_source_auto(
        times_hr,
        observed_drop_psi,
        source,
        observation,
        reservoir_template,
        robust_loss=robust_loss,
    )

    hmin = _minimum_rectangle_height(
        source,
        observation,
        aspect_ratio_width_over_height,
        source_xD,
        source_yD,
        rectangle_angle_deg,
    )
    h_starts = hmin * np.array([1.10, 1.5, 2.2, 3.2, 5.0, 8.0])

    best: ClosedRectangleFitResult | None = None
    for h0 in h_starts:
        if not np.isfinite(h0) or h0 <= 0:
            continue
        try:
            fit = fit_closed_rectangle(
                times_hr,
                observed_drop_psi,
                source,
                observation,
                reservoir_template,
                aspect_ratio_width_over_height=aspect_ratio_width_over_height,
                source_xD=source_xD,
                source_yD=source_yD,
                k0_md=float(base.k_md),
                phi_ct0_psi_inv=float(base.phi_ct_psi_inv),
                height0_ft=float(h0),
                rectangle_angle_deg=rectangle_angle_deg,
                robust_loss=robust_loss,
            )
        except Exception:
            continue
        if best is None or _candidate_score(fit) < _candidate_score(best):
            best = fit

    if best is None:
        raise RuntimeError("Automatic closed-rectangle matching failed for all internal start points.")
    return best


def fit_leaky_auto(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell,
    reservoir_template: Reservoir, robust_loss: str = "soft_l1",
):
    """Fit k, phi*ct and Hantush leakage factor without user start guesses."""
    from .advanced_fitting import fit_leaky_line_source
    base = fit_line_source_auto(
        times_hr, observed_drop_psi, source, observation, reservoir_template,
        robust_loss=robust_loss,
    )
    r = float(np.hypot(observation.x_ft-source.x_ft, observation.y_ft-source.y_ft))
    starts = r * np.array([0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    best = None
    for B0 in starts:
        try:
            fit = fit_leaky_line_source(
                times_hr, observed_drop_psi, source, observation, reservoir_template,
                float(base.k_md), float(base.phi_ct_psi_inv), float(max(B0, 1e-6)),
                robust_loss=robust_loss,
            )
        except Exception:
            continue
        if best is None or _candidate_score(fit) < _candidate_score(best):
            best = fit
    if best is None:
        raise RuntimeError("Automatic leaky-reservoir matching failed for all internal starts.")
    return best


def fit_double_porosity_auto(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell,
    reservoir_template: Reservoir, interporosity: str = "pss",
    robust_loss: str = "soft_l1", n_stehfest: int = 12,
):
    """Fit k_f, total phi*ct, omega and lambda with internal multistart."""
    from .advanced_fitting import fit_double_porosity
    if source.rw_ft is None or source.rw_ft <= 0:
        raise ValueError("Active-well rw_ft is required for the double-porosity model.")
    if interporosity not in {"pss", "transient_slab", "transient_sphere"}:
        raise ValueError("interporosity must be pss, transient_slab, or transient_sphere.")
    base = fit_line_source_auto(
        times_hr, observed_drop_psi, source, observation, reservoir_template,
        robust_loss=robust_loss,
    )
    omega_starts = (0.01, 0.05, 0.2, 0.5)
    lambda_starts = (1e-8, 1e-6, 1e-4, 1e-2)
    best = None
    for om0 in omega_starts:
        for lam0 in lambda_starts:
            try:
                fit = fit_double_porosity(
                    times_hr, observed_drop_psi, source, observation, reservoir_template,
                    rw_ft=float(source.rw_ft), k0_md=float(base.k_md),
                    phi_ct_total0_psi_inv=float(base.phi_ct_psi_inv),
                    omega0=om0, lam0=lam0, interporosity=interporosity,
                    robust_loss=robust_loss, n_stehfest=n_stehfest,
                )
            except Exception:
                continue
            if best is None or _candidate_score(fit) < _candidate_score(best):
                best = fit
    if best is None:
        raise RuntimeError("Automatic double-porosity matching failed for all internal starts.")
    return best


def fit_active_storage_skin_auto(
    times_hr: Sequence[float], observed_drop_psi: Sequence[float],
    source: SourceWell, observation: ObservationWell,
    reservoir_template: Reservoir, robust_loss: str = "soft_l1",
    n_stehfest: int = 12,
):
    """Fit k, phi*ct, active-well storage and skin without user start guesses."""
    from .advanced_fitting import fit_active_storage_skin
    if source.rw_ft is None or source.rw_ft <= 0:
        raise ValueError("Active-well rw_ft is required for active storage/skin fitting.")
    base = fit_line_source_auto(
        times_hr, observed_drop_psi, source, observation, reservoir_template,
        robust_loss=robust_loss,
    )
    CD_starts = (1e-2, 1.0, 1e2, 1e4, 1e6)
    skin_starts = (0.0, 2.0, 10.0)
    best = None
    for CD0 in CD_starts:
        for s0 in skin_starts:
            try:
                fit = fit_active_storage_skin(
                    times_hr, observed_drop_psi, source, observation, reservoir_template,
                    float(base.k_md), float(base.phi_ct_psi_inv), CD0, s0,
                    robust_loss=robust_loss, n_stehfest=n_stehfest,
                )
            except Exception:
                continue
            if best is None or _candidate_score(fit) < _candidate_score(best):
                best = fit
    if best is None:
        raise RuntimeError("Automatic active storage/skin matching failed for all internal starts.")
    return best
