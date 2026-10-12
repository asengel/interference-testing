"""Run manual simulations or automatic fits from readable CSV files.

Built-in case:     python examples/run.py earlougher_9_1
External case:     python examples/run.py --case-dir "C:/tests/synthetic_pulse"
Headless/batch:    python examples/run.py --all --no-show

One active and one observation well per CSV case. The library supports more.
"""
from __future__ import annotations

import argparse
import csv
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interference import (  # noqa: E402
    WELL_COLUMNS, RATE_COLUMNS, PRESSURE_COLUMNS,
    ClosedRectangle, LinearBoundary, ObservationWell, Reservoir, SourceWell,
    add_type_curve_match, dimensionless_pressure_constant_rate,
    dimensionless_time_over_distance_squared,
    fit_active_storage_skin_auto, fit_closed_rectangle_auto, fit_double_porosity_auto,
    fit_leaky_auto, fit_line_source_auto, fit_k_phi_ct,
    normalize_model_name, rate_levels_to_events,
    simulate_active_storage_skin, simulate_closed_rectangle,
    simulate_double_porosity, simulate_leaky_line_source, simulate_line_source,
    theis_well_function, validate_input_schema, validate_well_radius,
)
from interference.schema import canonicalize_inputs, MODEL_SPECS  # noqa: E402
from interference.auto_match import fit_single_linear_boundary  # noqa: E402
from interference.validation import (  # noqa: E402
    finite_number, positive_number, validate_fitted_parameters,
    validate_geometry, validate_history, validate_reservoir,
)

EXAMPLES = ROOT / "examples"
RESULTS = ROOT / "results"


def _read_inputs(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames not in (["parameter", "value"], ["parameter", "value", "description"]):
            raise ValueError(f"{path}: expected parameter,value[,description] columns.")
        values = {}
        for row in reader:
            key = (row.get("parameter") or "").strip()
            if not key or key.startswith("#"):
                continue
            if key in values:
                raise ValueError(f"{path}: duplicate parameter {key!r}.")
            values[key] = (row.get("value") or "").strip()
    validate_input_schema(values)
    return canonicalize_inputs(values)


def _read_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{path}: expected columns {','.join(columns)} in this order.")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _number(inputs: dict[str, str], key: str, default=None):
    val = inputs.get(key, "")
    if val:
        return finite_number(val, key)
    if default is not None:
        return float(default)
    raise ValueError(f"Missing required input '{key}'.")


def _optional_number(value: str):
    return None if value == "" else finite_number(value, "well radius")


def _boundary_from_inputs(inputs, source, observation):
    kind = inputs.get("boundary_type", "none").lower() or "none"
    if kind == "none":
        return ()
    d = positive_number(inputs["boundary_distance_ft"], "boundary_distance_ft")
    angle = finite_number(inputs["boundary_azimuth_deg"], "boundary_azimuth_deg")
    theta = np.deg2rad(angle)
    nx, ny = float(np.cos(theta)), float(np.sin(theta))
    # Azimuth is the normal from the active well TOWARD the boundary.
    boundary = LinearBoundary(source.x_ft + d * nx,
                              source.y_ft + d * ny, angle, kind)
    # Both wells must lie on the same physical reservoir side of the plane.
    observer_proj = ((observation.x_ft-source.x_ft) * nx
                     + (observation.y_ft-source.y_ft) * ny)
    if observer_proj >= d:
        raise ValueError(
            "Observation well lies on/across the specified boundary. Increase "
            "boundary_distance_ft or change boundary_azimuth_deg so both wells "
            "lie on the same reservoir side."
        )
    return (boundary,)


def _load_case(case_dir: Path):
    inputs = _read_inputs(case_dir / "inputs.csv")
    model = normalize_model_name(inputs["model"])
    wells = _read_rows(case_dir / "wells.csv", WELL_COLUMNS)
    if len(wells) < 2:
        raise ValueError("wells.csv needs one active and one observation well.")
    names = [w["well"] for w in wells]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Every well needs a unique, nonblank name.")
    active = [w for w in wells if w["role"].lower() in {"active", "source"}]
    observed = [w for w in wells if w["role"].lower() in {"observation", "obs"}]
    if len(active) != 1 or len(observed) != 1 or len(wells) != 2:
        raise ValueError("CSV runner supports exactly one active and one observation well.")
    aw, ow = active[0], observed[0]
    arw, orw = _optional_number(aw["rw_ft"]), _optional_number(ow["rw_ft"])
    validate_well_radius(model, arw, orw)

    rates = _read_rows(case_dir / "rates.csv", RATE_COLUMNS)
    if any(r["well"] not in names for r in rates):
        raise ValueError("rates.csv references an unknown well.")
    if any(r["well"] != aw["well"] for r in rates):
        raise ValueError("Only the active well may have a rate history in this CSV runner.")
    if not rates:
        raise ValueError("rates.csv needs at least one active well rate row.")
    t_rate = [r["time_hr"] for r in rates]
    q_rate = [r["rate_stbd"] for r in rates]
    t_rate, q_rate = validate_history(t_rate, q_rate, label="rates.csv")
    if not np.any(q_rate != 0):
        raise ValueError("rates.csv: all rate levels are zero; no interference signal.")

    mode = inputs.get("run_mode", "auto").strip().lower() or "auto"
    pressure_file = case_dir / "pressure.csv"
    if pressure_file.is_file():
        pressures = _read_rows(pressure_file, PRESSURE_COLUMNS)
    elif mode == "manual":
        pressures = []
    else:
        raise ValueError("auto mode requires pressure.csv with measured pressures.")
    if any(p["well"] != ow["well"] for p in pressures):
        raise ValueError("pressure.csv must contain data only for the observation well.")
    if pressures:
        t_obs, dp_obs = validate_history(
            [p["time_hr"] for p in pressures],
            [p["dp_psi"] for p in pressures],
            label="pressure.csv",
            min_points=(max(5, len(MODEL_SPECS[model].fitted_parameters) + 2)
                        if mode == "auto" else 1),
        )
    else:
        t_obs, dp_obs = np.array([], dtype=float), np.array([], dtype=float)
    if mode == "auto" and t_rate[0] > t_obs[0]:
        raise ValueError("First rate event occurs after the first pressure measurement.")

    source = SourceWell(
        aw["well"], _number(aw, "x_ft"), _number(aw, "y_ft"),
        rate_levels_to_events(t_rate, q_rate), arw,
    )
    observer = ObservationWell(
        ow["well"], _number(ow, "x_ft"), _number(ow, "y_ft"), orw,
    )
    # Automatic straight-boundary runs do not have a fixed wall location yet.
    # The fitting routine will construct its own trial image boundary.
    kind = inputs.get("boundary_type", "none").strip().lower() or "none"
    boundaries = (() if mode == "auto" and kind != "none"
                  else _boundary_from_inputs(inputs, source, observer))
    validate_geometry(source, observer, boundaries)

    h = _number(inputs, "h_ft")
    mu = _number(inputs, "mu_cp")
    B = _number(inputs, "B_rb_stb")
    ct = _number(inputs, "ct_psi_inv") if inputs.get("ct_psi_inv") else None
    validate_reservoir(h, mu, B, ct)
    template = Reservoir(h_ft=h, mu_cp=mu, B_rb_stb=B)
    if mode == "manual" and t_obs.size == 0 and not inputs.get("simulation_end_hr"):
        raise ValueError("Manual mode without pressure.csv requires simulation_end_hr in inputs.csv.")
    return inputs, model, t_obs, dp_obs, t_rate, q_rate, source, observer, template, boundaries


def _fit_auto_boundary(t, y, source, observer, template, kind, azimuth, loss,
                       distance_guess=None):
    """Fit permeability, storativity and boundary distance with fixed azimuth.

    A blank distance is supported. The initial distance guesses respect the
    condition that the active and observation wells lie on the same reservoir
    side of the wall, and multiple seeds reduce local-minimum sensitivity.
    """
    a = np.deg2rad(azimuth)
    nx, ny = float(np.cos(a)), float(np.sin(a))
    dx = observer.x_ft - source.x_ft
    dy = observer.y_ft - source.y_ft
    separation = float(np.hypot(dx, dy))
    projection = dx*nx + dy*ny
    dmin = max(1e-5, projection + max(abs(projection)*1e-5, 1e-5))
    distance_seeds = [max(dmin*1.1, max(projection, 0) + separation),
                      max(dmin*1.1, max(projection, 0) + separation*3)]
    if distance_guess is not None:
        if distance_guess <= dmin:
            raise ValueError(
                f"Initial boundary_distance_ft = {distance_guess:g} ft must exceed "
                f"{dmin:g} ft to keep both wells on the reservoir side of the boundary."
            )
        distance_seeds.insert(0, distance_guess)
    best = None
    last_error = None
    for d0 in distance_seeds:
        for k0 in (1, 100, 10000):
            for pct0 in (1e-9, 1e-7, 1e-5):
                try:
                    fit = fit_single_linear_boundary(
                        t, y, source, observer, template, kind, azimuth,
                        k0_md=k0, phi_ct0_psi_inv=pct0,
                        boundary_distance0_ft=d0, robust_loss=loss,
                    )
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    last_error = exc
                    continue
                if fit.success and np.isfinite(fit.rmse) and (best is None or fit.rmse < best.rmse):
                    best = fit
    if best is None:
        raise RuntimeError(
            "No converged boundary-distance fit. Check pressure sign, boundary "
            "azimuth, well geometry and whether the boundary response is observable."
        ) from last_error
    return best


def _fit_case(model, inputs, t, y, source, observer, template, boundaries):
    loss = inputs.get("robust_loss", "linear") or "linear"
    if model == "line_source":
        kind = inputs.get("boundary_type", "none").strip().lower() or "none"
        if kind != "none":
            azimuth = _number(inputs, "boundary_azimuth_deg")
            guess = (_number(inputs, "boundary_distance_ft")
                     if inputs.get("boundary_distance_ft") else None)
            fit = _fit_auto_boundary(t, y, source, observer, template, kind,
                                     azimuth, loss, distance_guess=guess)
            fitted_distance = fit.boundary_distance_ft
            a = np.deg2rad(azimuth)
            boundaries = (LinearBoundary(
                source.x_ft + fitted_distance*np.cos(a),
                source.y_ft + fitted_distance*np.sin(a),
                azimuth, kind),)
        else:
            fit = fit_line_source_auto(t, y, source, observer, template, robust_loss=loss)
        k, pct = fit.k_md, fit.phi_ct_psi_inv
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=pct)
        predict = lambda tt: simulate_line_source(tt, source, observer, k, res, boundaries)
        info = [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi"]
        if boundaries:
            info += [f"boundary = {boundaries[0].kind}",
                     f"fitted boundary distance = {fitted_distance:.6f} ft",
                     f"boundary azimuth (fixed) = {azimuth:.6g} deg"]

    elif model == "closed_rectangle":
        aspect = _number(inputs, "aspect_ratio_width_over_height")
        sx = _number(inputs, "source_xD")
        sy = _number(inputs, "source_yD")
        angle = _number(inputs, "rectangle_angle_deg", 0.0)
        fit = fit_closed_rectangle_auto(t, y, source, observer, template,
                                        aspect, sx, sy, rectangle_angle_deg=angle,
                                        robust_loss=loss)
        k, pct = fit.k_md, fit.phi_ct_psi_inv
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=pct)
        rect = ClosedRectangle(0.0, 0.0, fit.width_ft, fit.height_ft)
        local_src = SourceWell(source.name, sx*fit.width_ft, sy*fit.height_ft,
                               source.events, source.rw_ft)
        local_obs = ObservationWell(observer.name, fit.observation_xD*fit.width_ft,
                                    fit.observation_yD*fit.height_ft, observer.rw_ft)
        predict = lambda tt: simulate_closed_rectangle(tt, local_src, local_obs, k, res, rect)
        info = [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi",
                f"width = {fit.width_ft:.4f} ft", f"height = {fit.height_ft:.4f} ft",
                f"area = {fit.area_acres:.4f} acres",
                f"observation xD = {fit.observation_xD:.6f}",
                f"observation yD = {fit.observation_yD:.6f}"]

    elif model == "leaky":
        fit = fit_leaky_auto(t, y, source, observer, template, robust_loss=loss)
        k, pct = fit.k_md, fit.phi_ct_psi_inv
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=pct)
        predict = lambda tt: simulate_leaky_line_source(tt, source, observer, k, res, fit.leakage_factor_ft)
        info = [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi",
                f"leakage factor = {fit.leakage_factor_ft:.4f} ft"]

    elif model == "active_storage_skin":
        fit = fit_active_storage_skin_auto(t, y, source, observer, template, robust_loss=loss)
        k, pct = fit.k_md, fit.phi_ct_psi_inv
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=pct)
        predict = lambda tt: simulate_active_storage_skin(tt, source, observer, k, res,
                                                            fit.CD_active, fit.skin_active)
        info = [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi",
                f"CD active = {fit.CD_active:.6g}", f"skin active = {fit.skin_active:.4f}",
                f"C active = {fit.C_active_bbl_per_psi:.6g} bbl/psi"]

    elif model == "double_porosity":
        inter = inputs["interporosity_model"].lower()
        fit = fit_double_porosity_auto(t, y, source, observer, template, interporosity=inter,
                                        robust_loss=loss)
        k, pct = fit.k_fracture_md, fit.phi_ct_total_psi_inv
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=pct)
        predict = lambda tt: simulate_double_porosity(tt, source, observer, k, res,
            source.rw_ft, fit.omega, fit.lam, interporosity=inter)
        info = [f"k fracture = {k:.6f} mD", f"phi*ct total = {pct:.8e} 1/psi",
                f"omega = {fit.omega:.6g}", f"lambda = {fit.lam:.6g}",
                f"interporosity = {inter}"]
    else:
        raise ValueError(f"Unsupported model: {model}")

    ct = _number(inputs, "ct_psi_inv") if inputs.get("ct_psi_inv") else None
    validate_fitted_parameters(fit, model=model, compressibility=ct)
    if ct is not None:
        info.append(f"phi = {pct/ct:.6f}")
    info += [f"RMSE = {fit.rmse:.6f} psi"]
    return fit, predict, k, pct, info


def _manual_case(model, inputs, t, y, source, observer, template, boundaries):
    """Calculate pressure using user-supplied parameters; no optimizer is called."""
    k = _number(inputs, "permeability_md")
    pct = _number(inputs, "storativity_psi_inv")
    res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb,
                    phi_ct_psi_inv=pct)
    info = [f"k = {k:.6g} mD", f"phi*ct = {pct:.8e} 1/psi"]

    if model == "line_source":
        def predict(tt):
            return simulate_line_source(tt, source, observer, k, res, boundaries)
        if boundaries:
            info += [f"boundary = {boundaries[0].kind}",
                     f"boundary distance = {inputs['boundary_distance_ft']} ft",
                     f"boundary azimuth = {inputs['boundary_azimuth_deg']} deg"]

    elif model == "closed_rectangle":
        H = _number(inputs, "rectangle_height_ft")
        W = _number(inputs, "aspect_ratio_width_over_height") * H
        sx, sy = _number(inputs, "source_xD"), _number(inputs, "source_yD")
        angle = np.deg2rad(_number(inputs, "rectangle_angle_deg", 0))
        dxg, dyg = observer.x_ft-source.x_ft, observer.y_ft-source.y_ft
        dx = np.cos(angle)*dxg + np.sin(angle)*dyg
        dy = -np.sin(angle)*dxg + np.cos(angle)*dyg
        xs, ys = sx*W, sy*H
        xo, yo = xs+dx, ys+dy
        if not (0 < xo < W and 0 < yo < H):
            raise ValueError("The observation well lies outside the specified closed rectangle; "
                             "check rectangle_height_ft, orientation and well coordinates.")
        rectangle = ClosedRectangle(0.0, 0.0, W, H)
        local_src = SourceWell(source.name, xs, ys, source.events, source.rw_ft)
        local_obs = ObservationWell(observer.name, xo, yo, observer.rw_ft)
        def predict(tt):
            return simulate_closed_rectangle(tt, local_src, local_obs, k, res, rectangle)
        info += [f"width = {W:.6g} ft", f"height = {H:.6g} ft",
                 f"area = {W*H/43560:.6g} acres",
                 f"observation xD = {xo/W:.6f}", f"observation yD = {yo/H:.6f}"]

    elif model == "leaky":
        leak = _number(inputs, "leakage_factor_ft")
        def predict(tt):
            return simulate_leaky_line_source(tt, source, observer, k, res, leak)
        info.append(f"leakage factor = {leak:.6g} ft")

    elif model == "active_storage_skin":
        CD = _number(inputs, "active_well_storage_cd")
        skin = _number(inputs, "active_well_skin")
        def predict(tt):
            return simulate_active_storage_skin(tt, source, observer, k, res, CD, skin)
        C = CD * (2*np.pi*pct*template.h_ft*source.rw_ft**2) / 5.615
        info += [f"CD active = {CD:.6g}", f"skin active = {skin:.6g}",
                 f"C active = {C:.6g} bbl/psi"]

    elif model == "double_porosity":
        omega = _number(inputs, "fracture_storage_fraction")
        lam = _number(inputs, "interporosity_flow_parameter")
        inter = inputs["interporosity_model"].lower()
        def predict(tt):
            return simulate_double_porosity(tt, source, observer, k, res,
                source.rw_ft, omega, lam, interporosity=inter)
        info += [f"omega = {omega:.6g}", f"lambda = {lam:.6g}",
                 f"interporosity = {inter}"]

    else:
        raise ValueError(f"Unsupported model: {model}")

    p = np.asarray(predict(t), dtype=float) if len(t) else np.array([])
    if not np.all(np.isfinite(p)):
        raise ValueError("Manual model predicted nonfinite pressure values; check inputs.")
    residual = p-y
    rmse = float(np.sqrt(np.mean(residual**2))) if len(y) else None
    fit = SimpleNamespace(prediction=p, residual=residual, rmse=rmse, success=True)
    if inputs.get("ct_psi_inv"):
        ct = _number(inputs, "ct_psi_inv")
        if pct > ct:
            raise ValueError("storativity exceeds total compressibility, implying porosity > 1.")
        info.append(f"phi = {pct/ct:.6g}")
    info.append(f"RMSE = {rmse:.6f} psi" if rmse is not None else
                "RMSE = not available (no measured pressure)")
    return fit, predict, k, pct, info


def _simulation_times(inputs, t_obs, t_rate):
    """Dense prediction grid, with all rate-change times included exactly."""
    start = _number(inputs, "simulation_start_hr", 0.0)
    end_default = max(float(t_obs.max()) if len(t_obs) else 0.0,
                      float(t_rate.max())) * 1.06
    end = _number(inputs, "simulation_end_hr", end_default)
    if end <= start:
        raise ValueError("simulation_end_hr must be greater than simulation_start_hr.")
    n = int(_number(inputs, "simulation_points", 700))
    if len(t_obs) and (start > t_obs[0] or end < t_obs[-1]):
        raise ValueError("Simulation start/end must cover all measured pressure times.")
    if end < t_rate[0]:
        raise ValueError("Simulation end must be on/after first rate event.")
    # Include discontinuity times and observations exactly in exported series.
    inside = t_rate[(t_rate >= start) & (t_rate <= end)]
    obs_inside = t_obs[(t_obs >= start) & (t_obs <= end)]
    return np.unique(np.r_[np.linspace(start, end, n), inside, obs_inside])


def _make_plot(title, t_obs, dp_obs, t_rate, q_rate, source, observer,
               template, k, pct, predict, fit, out_png, *, mode="auto", tdense=None):
    """Four-panel interference-test diagnostic figure.

    The dimensionless panel uses the *first nonzero rate level* as the signed
    reference rate and time origin. For a pulse or multirate schedule, p_D is
    a rate-normalized superposed response, NOT a single-step Theis response.
    The Theis reference is included only as a constant-rate baseline.
    """
    if tdense is None:
        tdense = _simulation_times({}, t_obs, t_rate)
    t_obs = np.asarray(t_obs, dtype=float)
    dp_obs = np.asarray(dp_obs, dtype=float)
    tdense = np.asarray(tdense, dtype=float)
    t_rate = np.asarray(t_rate, dtype=float)
    q_rate = np.asarray(q_rate, dtype=float)
    p_curve = np.asarray(predict(tdense), dtype=float)
    if p_curve.shape != tdense.shape or not np.all(np.isfinite(p_curve)):
        raise ValueError("Model predicted invalid/nonfinite pressures on simulation time grid.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.3))
    ax_pressure, ax_type = axes[0]
    ax_resid, ax_rates = axes[1]
    curve_label = "Manual model" if mode == "manual" else "Matched model"

    # Panel A. Semilog time: exclude time=0 because log(0) is undefined.
    mask_curve = tdense > 0
    if np.any(mask_curve):
        ax_pressure.semilogx(tdense[mask_curve], p_curve[mask_curve],
                            color="tab:orange", lw=2, label=curve_label)
    mask_obs = t_obs > 0
    if np.any(mask_obs):
        ax_pressure.scatter(t_obs[mask_obs], dp_obs[mask_obs],
                            color="tab:blue", s=30, label="Observed", zorder=3)
    for event_time in t_rate[1:]:
        if event_time > 0 and event_time <= tdense.max():
            ax_pressure.axvline(event_time, ls=":", color="gray", alpha=0.45)
    ax_pressure.axhline(0, color="gray", lw=0.9, alpha=0.6)
    ax_pressure.set(xlabel="Elapsed time (hr, log scale)",
                    ylabel=r"Signed pressure drop $p_i-p$ (psi)",
                    title="A  |  Pressure response")
    ax_pressure.grid(True, which="both", alpha=0.23)
    ax_pressure.legend(loc="best")

    # Panel B. For any rate schedule, normalize the superposed response by
    # its first nonzero rate level. Time origin is that rate's start time.
    # Do NOT divide by an instantaneous rate: it would be undefined at shut-in.
    nonzero_rates = np.flatnonzero(q_rate != 0)
    if nonzero_rates.size == 0:
        raise ValueError("Cannot form dimensionless type curve: all rates are zero.")
    first = int(nonzero_rates[0])
    q_ref = float(q_rate[first])
    t_start = float(t_rate[first])
    res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb,
                    phi_ct_psi_inv=pct)
    distance = float(np.hypot(source.x_ft - observer.x_ft,
                              source.y_ft - observer.y_ft))
    if distance <= 0:
        raise ValueError("Type curve requires distinct active and observation wells.")
    u_curve = dimensionless_time_over_distance_squared(tdense - t_start, k, res, distance)
    pD_curve = dimensionless_pressure_constant_rate(p_curve, q_ref, k, res)
    u_obs = dimensionless_time_over_distance_squared(t_obs - t_start, k, res, distance)
    pD_obs = dimensionless_pressure_constant_rate(dp_obs, q_ref, k, res)

    # Type curves on log-log axes only show positive normalized pressure and time.
    valid_curve = np.isfinite(u_curve) & np.isfinite(pD_curve) & (u_curve > 0) & (pD_curve > 0)
    valid_obs = np.isfinite(u_obs) & np.isfinite(pD_obs) & (u_obs > 0) & (pD_obs > 0)
    valid_u = np.r_[u_curve[valid_curve], u_obs[valid_obs]]
    if valid_u.size:
        ref_u = np.logspace(np.log10(valid_u.min()), np.log10(valid_u.max()), 400)
        ref_pD = theis_well_function(ref_u)
    else:
        ref_u, ref_pD = np.array([], dtype=float), np.array([], dtype=float)

    add_type_curve_match(ax_type, u_curve, pD_curve, u_obs, pD_obs,
                         curve_label=curve_label, data_label="Observed (normalized)",
                         reference_u=ref_u, reference_pD=ref_pD,
                         reference_label="Theis (constant-rate reference)")
    # Zoom to the measured diagnostic window when observations are present;
    # very early Theis values may be >20 orders smaller than any observation.
    # A readable curve is more useful than an enormous log-y dynamic range.
    if np.any(valid_obs):
        lo_u, hi_u = float(u_obs[valid_obs].min()), float(u_obs[valid_obs].max())
        xlow, xhigh = lo_u * 0.75, hi_u * 1.15
    elif np.any(valid_curve):
        model_positive = pD_curve[valid_curve]
        informative = valid_curve & (pD_curve >= model_positive.max() * 1e-5)
        xlow = float(u_curve[informative].min()) * 0.8
        xhigh = float(u_curve[valid_curve].max()) * 1.15
    else:
        xlow = xhigh = None
    if xlow is not None and xhigh > xlow > 0:
        ax_type.set_xlim(xlow, xhigh)
        keep_curve = valid_curve & (u_curve >= xlow) & (u_curve <= xhigh)
        y_focus = np.r_[pD_curve[keep_curve], pD_obs[valid_obs]]
        if y_focus.size:
            ymax = float(y_focus.max())
            ymin = max(float(y_focus.min()) * 0.3, ymax * 1e-7)
            ax_type.set_ylim(ymin, ymax * 2.5)
    ax_type.set_title(r"B  |  Dimensionless response: $p_D$ vs $t_D/r_D^2$")
    # Theis remains a reference after shut-in, NOT the expected pulse shape.
    variable_rate = len(q_rate) != 1 or first != 0
    if variable_rate:
        ax_type.text(0.03, 0.04,
                     "Pulse / variable rate: normalized superposition, not a single-rate type curve",
                     transform=ax_type.transAxes, fontsize=8, ha="left", va="bottom",
                     bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    # Only flag actually unplottable *pressure* responses. The zero-time
    # origin is omitted on logarithmic axes by definition, not a data issue.
    omitted = int(np.count_nonzero((u_curve > 0) & ~valid_curve)
                  + np.count_nonzero((u_obs > 0) & ~valid_obs))
    if omitted:
        ax_type.text(0.03, 0.96, f"{omitted} nonpositive p_D values omitted (log scale)",
                     transform=ax_type.transAxes, fontsize=8, va="top",
                     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    if not np.any(valid_curve) and not np.any(valid_obs):
        ax_type.text(0.5, 0.5, "No positive dimensionless pressure values",
                     transform=ax_type.transAxes, ha="center", va="center")

    # Panel C. Residuals have the intuitive observed minus calculated sign.
    ax_resid.axhline(0, lw=1.2, color="gray", alpha=0.8)
    if t_obs.size:
        obs_minus_calc = dp_obs - np.asarray(fit.prediction, dtype=float)
        ax_resid.scatter(t_obs, obs_minus_calc, color="tab:blue", s=30, zorder=3)
        ax_resid.set_ylabel(r"Observed $-$ calculated (psi)")
    else:
        ax_resid.text(0.5, 0.5, "No measured pressure data — residuals unavailable",
                      transform=ax_resid.transAxes, ha="center", va="center")
        ax_resid.set_ylabel("Residual (psi)")
    ax_resid.set(xlabel="Elapsed time (hr)", title="C  |  Pressure residuals")
    ax_resid.grid(True, alpha=0.23)

    # Panel D. Rate *levels* must be drawn as a post-step schedule; a pulse
    # therefore visibly drops to zero at shut-in and stays there.
    rate_end = float(tdense.max())
    step_time = np.r_[t_rate, max(rate_end, float(t_rate[-1]))]
    step_rates = np.r_[q_rate, q_rate[-1]]
    ax_rates.step(step_time, step_rates, where="post", lw=2,
                  color="tab:green", label="Active well")
    ax_rates.axhline(0, color="gray", lw=0.9)
    ax_rates.set(xlabel="Elapsed time (hr)", ylabel="Rate (STB/day)",
                 title="D  |  Production (+) / injection (−) rate")
    ax_rates.set_xlim(float(tdense.min()), rate_end)
    ax_rates.grid(True, alpha=0.23)
    ax_rates.legend(loc="best")

    score = f"RMSE={fit.rmse:.3g} psi" if fit.rmse is not None else "no observations"
    fig.suptitle(f"{title}  |  {mode.upper()}  |  k={k:.4g} mD, "
                 f"phi*ct={pct:.3g} 1/psi  |  {score}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.3, w_pad=2.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    return fig

def run_directory(case_dir: Path, *, show: bool = True, result_dir: Path | None = None):
    case_dir = case_dir.expanduser().resolve()
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    inputs, model, t, y, t_rate, rates, source, observer, template, boundary = _load_case(case_dir)
    mode = inputs.get("run_mode", "auto").strip().lower() or "auto"
    if mode == "manual":
        fit, predict, k, pct, lines = _manual_case(
            model, inputs, t, y, source, observer, template, boundary)
    else:
        fit, predict, k, pct, lines = _fit_case(
            model, inputs, t, y, source, observer, template, boundary)
    name = inputs.get("case_name") or case_dir.name
    result_dir = result_dir or (RESULTS if case_dir.parent.resolve() == EXAMPLES.resolve()
                                 else case_dir / "results")
    result_dir.mkdir(parents=True, exist_ok=True)
    png_path = result_dir / f"{case_dir.name}.png"
    txt_path = result_dir / f"{case_dir.name}.txt"
    simulation_times = _simulation_times(inputs, t, t_rate)
    predicted = np.asarray(predict(simulation_times), float)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Model predicted nonfinite pressures on simulation time grid.")
    fig = _make_plot(name, t, y, t_rate, rates, source, observer, template,
                     k, pct, predict, fit, png_path, mode=mode, tdense=simulation_times)
    csv_path = result_dir / f"{case_dir.name}_predicted.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_hr", "dp_psi"])
        writer.writerows(zip(simulation_times, predicted))
    summary = "\n".join([name, f"model = {model}", f"run_mode = {mode}", *lines]) + "\n"
    txt_path.write_text(summary, encoding="utf-8")
    print(f"\n[{case_dir.name}]\n{summary.rstrip()}\nfigure: {png_path}\npredicted pressures: {csv_path}\nresults: {txt_path}", flush=True)
    try:
        if show:
            import matplotlib
            backend = str(matplotlib.get_backend()).lower()
            if backend in {"agg", "pdf", "ps", "svg", "cairo"}:
                # A terminal configured with a non-GUI backend cannot show
                # Matplotlib windows. Open the saved PNG in the OS viewer.
                if sys.platform == "win32":
                    import os
                    os.startfile(str(png_path))
                else:
                    import webbrowser
                    webbrowser.open(png_path.resolve().as_uri())
            else:
                plt.show()  # Native plot window, or inline with an interactive backend.
    finally:
        plt.close(fig)
    return png_path, txt_path, summary


def run_case(case_name: str, *, show: bool = True):
    return run_directory(EXAMPLES / case_name, show=show, result_dir=RESULTS)


def available_cases() -> list[str]:
    if not EXAMPLES.exists():
        return []
    return sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir()
                  and all((p / filename).exists() for filename in
                          ("inputs.csv", "wells.csv", "rates.csv")))


def main():
    parser = argparse.ArgumentParser(description="Run manual or automatic interference tests from CSV files.")
    parser.add_argument("case", nargs="?", help="Case folder name inside examples/")
    parser.add_argument("--case-dir", type=Path, help="External folder with inputs/wells/rates CSV and optional pressure CSV for manual mode")
    parser.add_argument("--all", action="store_true", help="Run built-in examples")
    parser.add_argument("--list", action="store_true", help="List built-in examples")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening plot windows")
    args = parser.parse_args()
    if args.list:
        print("\n".join(available_cases()))
        return
    if sum(bool(x) for x in (args.case, args.case_dir, args.all)) != 1:
        parser.error("Supply a case name, --case-dir, or --all (exactly one).")
    if args.case_dir:
        run_directory(args.case_dir, show=not args.no_show)
    elif args.all:
        for name in available_cases():
            run_case(name, show=not args.no_show)
    else:
        run_case(args.case, show=not args.no_show)


if __name__ == "__main__":
    main()
