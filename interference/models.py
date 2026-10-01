from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

from .kernels import theis_well_function, sandal_active_storage_skin, tongpenyai_two_well

FIELD_TIME = 0.0002637
FIELD_PRESSURE = 141.2


@dataclass(frozen=True)
class Reservoir:
    h_ft: float
    mu_cp: float
    B_rb_stb: float = 1.0
    phi: float | None = None
    ct_psi_inv: float | None = None
    phi_ct_psi_inv: float | None = None

    @property
    def phi_ct(self) -> float:
        if self.phi_ct_psi_inv is not None:
            return float(self.phi_ct_psi_inv)
        if self.phi is None or self.ct_psi_inv is None:
            raise ValueError("Provide either phi_ct_psi_inv or both phi and ct_psi_inv.")
        return float(self.phi * self.ct_psi_inv)


@dataclass(frozen=True)
class RateEvent:
    time_hr: float
    delta_q_stbd: float  # production positive; injection negative


@dataclass(frozen=True)
class SourceWell:
    name: str
    x_ft: float
    y_ft: float
    events: tuple[RateEvent, ...]
    rw_ft: float | None = None

    def __post_init__(self):
        if self.rw_ft is not None and self.rw_ft <= 0:
            raise ValueError("rw_ft must be positive when supplied.")


@dataclass(frozen=True)
class ObservationWell:
    name: str
    x_ft: float
    y_ft: float
    rw_ft: float | None = None

    def __post_init__(self):
        if self.rw_ft is not None and self.rw_ft <= 0:
            raise ValueError("rw_ft must be positive when supplied.")


@dataclass(frozen=True)
class LinearBoundary:
    """Infinite straight boundary represented by a point and unit-normal angle.

    kind: 'no_flow' -> same-sign image; 'constant_pressure' -> opposite-sign image.
    angle_deg is the normal-vector azimuth measured CCW from +x.
    """
    x0_ft: float
    y0_ft: float
    angle_deg: float
    kind: str = "no_flow"

    @property
    def image_sign(self) -> float:
        if self.kind == "no_flow":
            return 1.0
        if self.kind == "constant_pressure":
            return -1.0
        raise ValueError("Boundary kind must be 'no_flow' or 'constant_pressure'.")


def rate_levels_to_events(times_hr: Sequence[float], rates_stbd: Sequence[float]) -> tuple[RateEvent, ...]:
    """Convert piecewise-constant rate levels to rate-change events.

    times_hr[i] is when rates_stbd[i] becomes active. The pre-first-event rate is zero.
    """
    t = np.asarray(times_hr, dtype=float)
    q = np.asarray(rates_stbd, dtype=float)
    if t.ndim != 1 or q.ndim != 1 or t.size != q.size or t.size == 0:
        raise ValueError("times_hr and rates_stbd must be same-length 1D nonempty arrays.")
    if np.any(np.diff(t) < 0):
        raise ValueError("Rate times must be nondecreasing.")
    dq = np.diff(np.r_[0.0, q])
    return tuple(RateEvent(float(tt), float(dd)) for tt, dd in zip(t, dq) if dd != 0.0)


def pulse_events(start_hr: float, stop_hr: float, q_stbd: float) -> tuple[RateEvent, ...]:
    if stop_hr <= start_hr:
        raise ValueError("stop_hr must exceed start_hr.")
    return (RateEvent(start_hr, q_stbd), RateEvent(stop_hr, -q_stbd))


def _mirror_point(x: float, y: float, b: LinearBoundary) -> tuple[float, float]:
    a = np.deg2rad(b.angle_deg)
    nx, ny = np.cos(a), np.sin(a)
    dx, dy = x - b.x0_ft, y - b.y0_ft
    signed = dx * nx + dy * ny
    return x - 2.0 * signed * nx, y - 2.0 * signed * ny


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _isotropic_step_drop(
    dt_hr: np.ndarray,
    dq_stbd: float,
    r_ft: float,
    k_md: float,
    reservoir: Reservoir,
) -> np.ndarray:
    if r_ft <= 0 or k_md <= 0:
        raise ValueError("r_ft and k_md must be positive.")
    u = FIELD_TIME * k_md * dt_hr / (reservoir.mu_cp * reservoir.phi_ct * r_ft**2)
    pD = theis_well_function(u)
    return FIELD_PRESSURE * dq_stbd * reservoir.B_rb_stb * reservoir.mu_cp / (k_md * reservoir.h_ft) * pD


def simulate_line_source(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    k_md: float,
    reservoir: Reservoir,
    boundaries: Sequence[LinearBoundary] = (),
) -> np.ndarray:
    """Signed pressure drop p_i-p at an observation well.

    Production is positive rate and gives positive pressure drop. Injection is negative
    and gives negative pressure drop (pressure rise).
    """
    t = np.asarray(times_hr, dtype=float)
    out = np.zeros_like(t)
    direct_r = _distance((source.x_ft, source.y_ft), (observation.x_ft, observation.y_ft))
    for ev in source.events:
        dt = t - ev.time_hr
        out += _isotropic_step_drop(dt, ev.delta_q_stbd, direct_r, k_md, reservoir)
        for b in boundaries:
            ix, iy = _mirror_point(source.x_ft, source.y_ft, b)
            image_r = _distance((ix, iy), (observation.x_ft, observation.y_ft))
            out += b.image_sign * _isotropic_step_drop(dt, ev.delta_q_stbd, image_r, k_md, reservoir)
    return out


def simulate_multiwell(
    times_hr: Sequence[float],
    sources: Sequence[SourceWell],
    observation: ObservationWell,
    k_md: float,
    reservoir: Reservoir,
    boundaries: Sequence[LinearBoundary] = (),
) -> np.ndarray:
    out = np.zeros(len(times_hr), dtype=float)
    for src in sources:
        out += simulate_line_source(times_hr, src, observation, k_md, reservoir, boundaries)
    return out


def simulate_anisotropic(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    kx_md: float,
    ky_md: float,
    theta_deg: float,
    reservoir: Reservoir,
) -> np.ndarray:
    """Homogeneous 2D permeability anisotropy using coordinate transformation.

    kx, ky are principal permeabilities and theta is the principal-x azimuth CCW from +x.
    Equivalent flow capacity is sqrt(kx*ky)*h.
    """
    if kx_md <= 0 or ky_md <= 0:
        raise ValueError("Principal permeabilities must be positive.")
    dx = observation.x_ft - source.x_ft
    dy = observation.y_ft - source.y_ft
    a = np.deg2rad(theta_deg)
    xp = np.cos(a) * dx + np.sin(a) * dy
    yp = -np.sin(a) * dx + np.cos(a) * dy
    k_eq = float(np.sqrt(kx_md * ky_md))
    r_eq_sq = k_eq * (xp * xp / kx_md + yp * yp / ky_md)
    if r_eq_sq <= 0:
        raise ValueError("Observation point must differ from source point.")
    r_eq = float(np.sqrt(r_eq_sq))
    t = np.asarray(times_hr, dtype=float)
    out = np.zeros_like(t)
    for ev in source.events:
        dt = t - ev.time_hr
        out += _isotropic_step_drop(dt, ev.delta_q_stbd, r_eq, k_eq, reservoir)
    return out


def simulate_active_storage_skin_step(
    times_hr: Sequence[float],
    q_stbd: float,
    r_ft: float,
    rw_ft: float,
    k_md: float,
    reservoir: Reservoir,
    CD_active: float,
    skin_active: float,
    start_hr: float = 0.0,
    n_stehfest: int = 12,
) -> np.ndarray:
    """Single rate step with active-well storage and skin (Sandal model)."""
    t = np.asarray(times_hr, dtype=float)
    dt = t - start_hr
    tD = np.where(dt > 0, FIELD_TIME * k_md * dt / (reservoir.mu_cp * reservoir.phi_ct * rw_ft**2), 0.0)
    rD = r_ft / rw_ft
    pD = sandal_active_storage_skin(tD, rD, CD_active, skin_active, n_stehfest)
    return FIELD_PRESSURE * q_stbd * reservoir.B_rb_stb * reservoir.mu_cp / (k_md * reservoir.h_ft) * pD


def simulate_active_storage_skin(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    k_md: float,
    reservoir: Reservoir,
    CD_active: float,
    skin_active: float,
    n_stehfest: int = 12,
) -> np.ndarray:
    """Arbitrary surface-rate history with storage/skin at the active well.

    Linear superposition is applied to each surface-rate change. The active-well
    radius is taken from ``source.rw_ft``.
    """
    if source.rw_ft is None:
        raise ValueError("Active-well rw_ft is required for the active storage/skin model.")
    r = _distance((source.x_ft, source.y_ft), (observation.x_ft, observation.y_ft))
    out = np.zeros(len(times_hr), dtype=float)
    for ev in source.events:
        out += simulate_active_storage_skin_step(
            times_hr, ev.delta_q_stbd, r, source.rw_ft, k_md, reservoir,
            CD_active, skin_active, start_hr=ev.time_hr, n_stehfest=n_stehfest,
        )
    return out


def simulate_two_well_storage_skin_step(
    times_hr: Sequence[float],
    q_stbd: float,
    r_ft: float,
    rw_ft: float,
    k_md: float,
    reservoir: Reservoir,
    CD_obs: float,
    skin_obs: float,
    CD_active: float,
    skin_active: float,
    start_hr: float = 0.0,
    n_stehfest: int = 12,
    n_quad: int = 64,
) -> np.ndarray:
    """Single rate step with storage/skin at both wells (Tongpenyai model)."""
    t = np.asarray(times_hr, dtype=float)
    dt = t - start_hr
    tD = np.where(dt > 0, FIELD_TIME * k_md * dt / (reservoir.mu_cp * reservoir.phi_ct * rw_ft**2), 0.0)
    rD = r_ft / rw_ft
    pD = tongpenyai_two_well(
        tD, rD, CD_obs, skin_obs, CD_active, skin_active, n_stehfest, n_quad
    )
    return FIELD_PRESSURE * q_stbd * reservoir.B_rb_stb * reservoir.mu_cp / (k_md * reservoir.h_ft) * pD


@dataclass(frozen=True)
class ClosedRectangle:
    """Closed (no-flow) rectangular reservoir aligned with the model x-y axes."""
    x_min_ft: float
    y_min_ft: float
    width_ft: float
    height_ft: float

    def __post_init__(self):
        if self.width_ft <= 0 or self.height_ft <= 0:
            raise ValueError("Rectangle width and height must be positive.")

    @property
    def area_ft2(self) -> float:
        return float(self.width_ft * self.height_ft)


# Import advanced kernels late to keep the core imports readable.
from .kernels import hantush_well_function, double_porosity_line_source


def _rectangle_auto_layers(k_md: float, reservoir: Reservoir, tmax_hr: float,
                           width_ft: float, height_ft: float) -> int:
    """Heuristic image-lattice truncation for a closed rectangle."""
    if tmax_hr <= 0:
        return 2
    diffusivity = FIELD_TIME * k_md / (reservoir.mu_cp * reservoir.phi_ct)  # ft2/hr
    # E1(x) is negligible once x=r^2/(4Dt) is large. r_max uses x~1/30.
    rmax = np.sqrt(max(120.0 * diffusivity * tmax_hr, 0.0))
    n = int(np.ceil(rmax / (2.0 * min(width_ft, height_ft)))) + 2
    return int(np.clip(n, 2, 50))


def _closed_rectangle_step_drop(
    dt_hr: np.ndarray,
    dq_stbd: float,
    source_xy: tuple[float, float],
    observation_xy: tuple[float, float],
    rectangle: ClosedRectangle,
    k_md: float,
    reservoir: Reservoir,
    image_layers: int | None = None,
) -> np.ndarray:
    """One rate-step response in a closed rectangle using the image lattice."""
    t = np.asarray(dt_hr, dtype=float)
    out = np.zeros_like(t)
    if not np.any(t > 0):
        return out

    W, H = rectangle.width_ft, rectangle.height_ft
    xs = source_xy[0] - rectangle.x_min_ft
    ys = source_xy[1] - rectangle.y_min_ft
    xo = observation_xy[0] - rectangle.x_min_ft
    yo = observation_xy[1] - rectangle.y_min_ft
    if not (0 <= xs <= W and 0 <= ys <= H and 0 <= xo <= W and 0 <= yo <= H):
        raise ValueError("Source and observation wells must lie inside the closed rectangle.")

    M = image_layers
    if M is None:
        M = _rectangle_auto_layers(k_md, reservoir, float(np.max(t)), W, H)
    if M < 0:
        raise ValueError("image_layers must be nonnegative.")

    pref = FIELD_PRESSURE * dq_stbd * reservoir.B_rb_stb * reservoir.mu_cp / (k_md * reservoir.h_ft)
    mask = t > 0
    tp = t[mask]
    total = np.zeros_like(tp)

    # Neumann/no-flow boundaries: all reflected image sources have the same sign.
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            for xi in (2.0*m*W + xs, 2.0*m*W - xs):
                for yi in (2.0*n*H + ys, 2.0*n*H - ys):
                    r2 = (xo - xi)**2 + (yo - yi)**2
                    if r2 <= 0.0:
                        # An observation well coincident with the active well requires a
                        # finite-radius well solution, not the line-source interference kernel.
                        raise ValueError("Closed-rectangle line-source model cannot evaluate at the active well.")
                    u = FIELD_TIME * k_md * tp / (reservoir.mu_cp * reservoir.phi_ct * r2)
                    total += theis_well_function(u)
    out[mask] = pref * total
    return out


def simulate_closed_rectangle(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    k_md: float,
    reservoir: Reservoir,
    rectangle: ClosedRectangle,
    image_layers: int | None = None,
) -> np.ndarray:
    """Pressure-drop history for arbitrary rate changes in a closed rectangle."""
    t = np.asarray(times_hr, dtype=float)
    out = np.zeros_like(t)
    sxy = (source.x_ft, source.y_ft)
    oxy = (observation.x_ft, observation.y_ft)
    for ev in source.events:
        out += _closed_rectangle_step_drop(
            t - ev.time_hr, ev.delta_q_stbd, sxy, oxy,
            rectangle, k_md, reservoir, image_layers
        )
    return out


def simulate_leaky_line_source(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    k_md: float,
    reservoir: Reservoir,
    leakage_factor_ft: float,
) -> np.ndarray:
    """Hantush-Jacob leaky-reservoir interference response.

    leakage_factor_ft corresponds to B in the leaky-aquifer formulation.  As
    B -> infinity, the response approaches the Theis line-source model.
    """
    if leakage_factor_ft <= 0:
        raise ValueError("leakage_factor_ft must be positive.")
    t = np.asarray(times_hr, dtype=float)
    r = _distance((source.x_ft, source.y_ft), (observation.x_ft, observation.y_ft))
    if r <= 0:
        raise ValueError("Observation point must differ from source point.")
    beta = r / leakage_factor_ft
    pref = FIELD_PRESSURE * reservoir.B_rb_stb * reservoir.mu_cp / (k_md * reservoir.h_ft)
    out = np.zeros_like(t)
    for ev in source.events:
        dt = t - ev.time_hr
        mask = dt > 0
        if not np.any(mask):
            continue
        u_classic = (reservoir.mu_cp * reservoir.phi_ct * r*r) / (
            4.0 * FIELD_TIME * k_md * dt[mask]
        )
        pD = 0.5 * hantush_well_function(u_classic, beta)
        out[mask] += pref * ev.delta_q_stbd * pD
    return out


def simulate_double_porosity(
    times_hr: Sequence[float],
    source: SourceWell,
    observation: ObservationWell,
    k_fracture_md: float,
    reservoir_total: Reservoir,
    rw_ft: float,
    omega: float,
    lam: float,
    interporosity: str = "pss",
    n_stehfest: int = 12,
) -> np.ndarray:
    """Constant-property double-porosity interference with arbitrary rate history.

    The flowing permeability is the fissure/high-mobility-medium permeability.
    reservoir_total.phi_ct is the total (fissure + matrix) storativity per unit
    bulk volume. omega is fissure storativity / total storativity and lambda is
    the dimensionless interporosity-flow parameter of Deruyck/Warren-Root.
    """
    if rw_ft <= 0 or k_fracture_md <= 0:
        raise ValueError("rw_ft and k_fracture_md must be positive.")
    r = _distance((source.x_ft, source.y_ft), (observation.x_ft, observation.y_ft))
    rD = r / rw_ft
    t = np.asarray(times_hr, dtype=float)
    out = np.zeros_like(t)
    pref = FIELD_PRESSURE * reservoir_total.B_rb_stb * reservoir_total.mu_cp / (
        k_fracture_md * reservoir_total.h_ft
    )
    for ev in source.events:
        dt = t - ev.time_hr
        tD = np.where(
            dt > 0,
            FIELD_TIME * k_fracture_md * dt /
            (reservoir_total.mu_cp * reservoir_total.phi_ct * rw_ft**2),
            0.0,
        )
        pD = double_porosity_line_source(
            tD, rD, omega, lam, interporosity=interporosity,
            n_stehfest=n_stehfest,
        )
        out += pref * ev.delta_q_stbd * pD
    return out
