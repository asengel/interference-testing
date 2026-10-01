from __future__ import annotations

from typing import Sequence
import numpy as np

from .models import FIELD_TIME, FIELD_PRESSURE, Reservoir


def dimensionless_time_over_distance_squared(
    times_hr: Sequence[float] | np.ndarray,
    k_md: float,
    reservoir: Reservoir,
    r_ft: float,
) -> np.ndarray:
    """Return t_D/r_D^2 for the field-unit convention used by this package."""
    t = np.asarray(times_hr, dtype=float)
    if k_md <= 0 or r_ft <= 0:
        raise ValueError("k_md and r_ft must be positive.")
    return FIELD_TIME * k_md * t / (reservoir.mu_cp * reservoir.phi_ct * r_ft**2)


def dimensionless_pressure_constant_rate(
    pressure_drop_psi: Sequence[float] | np.ndarray,
    q_stbd: float,
    k_md: float,
    reservoir: Reservoir,
) -> np.ndarray:
    """Convert pressure drop to p_D for a single constant-rate interference step.

    The package sign convention is production positive, injection negative, and
    pressure drop p_i-p has the same sign as q. Therefore p_D remains positive
    for either production or injection as long as the supplied q is the active
    constant-rate step used for the type-curve scaling.
    """
    if q_stbd == 0 or k_md <= 0:
        raise ValueError("q_stbd must be nonzero and k_md must be positive.")
    dp = np.asarray(pressure_drop_psi, dtype=float)
    return (
        dp * k_md * reservoir.h_ft
        / (FIELD_PRESSURE * q_stbd * reservoir.B_rb_stb * reservoir.mu_cp)
    )


def add_type_curve_match(
    ax,
    curve_u: Sequence[float] | np.ndarray,
    curve_pD: Sequence[float] | np.ndarray,
    data_u: Sequence[float] | np.ndarray,
    data_pD: Sequence[float] | np.ndarray,
    *,
    curve_label: str = "model type curve",
    data_label: str = "scaled data",
    reference_u: Sequence[float] | np.ndarray | None = None,
    reference_pD: Sequence[float] | np.ndarray | None = None,
    reference_label: str = "reference",
) -> None:
    """Plot a standard log-log interference type-curve match on an axes object."""
    cu = np.asarray(curve_u, dtype=float)
    cp = np.asarray(curve_pD, dtype=float)
    du = np.asarray(data_u, dtype=float)
    dp = np.asarray(data_pD, dtype=float)

    mask_c = (cu > 0) & (cp > 0) & np.isfinite(cu) & np.isfinite(cp)
    mask_d = (du > 0) & (dp > 0) & np.isfinite(du) & np.isfinite(dp)
    ax.loglog(cu[mask_c], cp[mask_c], label=curve_label)
    ax.scatter(du[mask_d], dp[mask_d], label=data_label, zorder=3)

    if reference_u is not None and reference_pD is not None:
        ru = np.asarray(reference_u, dtype=float)
        rp = np.asarray(reference_pD, dtype=float)
        mask_r = (ru > 0) & (rp > 0) & np.isfinite(ru) & np.isfinite(rp)
        ax.loglog(ru[mask_r], rp[mask_r], linestyle="--", label=reference_label)

    ax.set_xlabel(r"$t_D/r_D^2$")
    ax.set_ylabel(r"$p_D$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
