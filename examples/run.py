"""Run CSV-defined interference-test cases.

User-facing files in each case folder
-------------------------------------
inputs.csv     reservoir/fluid settings + selected model
wells.csv      well geometry and optional wellbore radius
rates.csv      rate levels by active well
pressure.csv   measured pressure change by observation well

Unknown hydraulic/model parameters are fitted; they are not supplied as guesses.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interference import (  # noqa: E402
    INPUT_SCHEMA, WELL_COLUMNS, RATE_COLUMNS, PRESSURE_COLUMNS,
    ClosedRectangle, ObservationWell, Reservoir, SourceWell,
    add_type_curve_match, dimensionless_pressure_constant_rate,
    dimensionless_time_over_distance_squared,
    fit_active_storage_skin_auto, fit_closed_rectangle_auto,
    fit_double_porosity_auto, fit_leaky_auto, fit_line_source_auto,
    normalize_model_name, rate_levels_to_events,
    simulate_active_storage_skin, simulate_closed_rectangle,
    simulate_double_porosity, simulate_leaky_line_source, simulate_line_source,
    theis_well_function, validate_input_schema, validate_well_radius,
)

EXAMPLES = ROOT / "examples"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _read_inputs(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        if rows.fieldnames != ["parameter", "value"]:
            raise ValueError(f"{path} must contain exactly the columns: parameter,value")
        out: dict[str, str] = {}
        for row in rows:
            key = (row.get("parameter") or "").strip()
            if key:
                out[key] = (row.get("value") or "").strip()
    validate_input_schema(out)
    # Enforce one stable row schema across every case, including unused blank rows.
    extra = sorted(set(out) - set(INPUT_SCHEMA))
    if extra:
        raise ValueError(f"{path} contains non-schema rows: {', '.join(extra)}")
    return out


def _read_rows(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != required_columns:
            raise ValueError(
                f"{path} must use columns {','.join(required_columns)} in that order; "
                f"found {','.join(reader.fieldnames or [])}."
            )
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _f(s: dict[str, str], key: str, default: float | None = None) -> float:
    if s.get(key, "") != "":
        return float(s[key])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing required input for selected model: {key}")


def _optional_f(value: str) -> float | None:
    value = value.strip()
    return None if value == "" else float(value)


def _s(s: dict[str, str], key: str, default: str | None = None) -> str:
    if s.get(key, "") != "":
        return s[key]
    if default is not None:
        return default
    raise KeyError(f"Missing required input for selected model: {key}")


def _load_case(case_name: str):
    case_dir = EXAMPLES / case_name
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Example case not found: {case_dir}")

    inputs = _read_inputs(case_dir / "inputs.csv")
    model = normalize_model_name(_s(inputs, "model"))

    well_rows = _read_rows(case_dir / "wells.csv", WELL_COLUMNS)
    if not well_rows:
        raise ValueError("wells.csv must contain at least one active and one observation well.")
    names = [r["well"] for r in well_rows]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("Every well in wells.csv must have a unique nonblank name.")

    active_rows = [r for r in well_rows if r["role"].lower() in {"active", "source"}]
    obs_rows = [r for r in well_rows if r["role"].lower() in {"observation", "obs"}]
    if len(active_rows) != 1 or len(obs_rows) != 1:
        raise ValueError(
            "The current CSV runner supports one active and one observation well per case. "
            "wells.csv is already named/structured for later multiwell extension."
        )
    ar, orow = active_rows[0], obs_rows[0]
    active_rw = _optional_f(ar["rw_ft"])
    obs_rw = _optional_f(orow["rw_ft"])
    validate_well_radius(model, active_rw, obs_rw)

    rate_rows = _read_rows(case_dir / "rates.csv", RATE_COLUMNS)
    unknown_rate_wells = sorted({r["well"] for r in rate_rows} - set(names))
    if unknown_rate_wells:
        raise ValueError(f"rates.csv refers to unknown wells: {', '.join(unknown_rate_wells)}")
    active_rate_rows = [r for r in rate_rows if r["well"] == ar["well"]]
    if not active_rate_rows:
        raise ValueError(f"rates.csv contains no rate history for active well {ar['well']!r}.")
    t_rate = np.array([float(r["time_hr"]) for r in active_rate_rows], float)
    q_level = np.array([float(r["rate_stbd"]) for r in active_rate_rows], float)

    pressure_rows = _read_rows(case_dir / "pressure.csv", PRESSURE_COLUMNS)
    unknown_pressure_wells = sorted({r["well"] for r in pressure_rows} - set(names))
    if unknown_pressure_wells:
        raise ValueError(f"pressure.csv refers to unknown wells: {', '.join(unknown_pressure_wells)}")
    obs_pressure_rows = [r for r in pressure_rows if r["well"] == orow["well"]]
    if not obs_pressure_rows:
        raise ValueError(f"pressure.csv contains no data for observation well {orow['well']!r}.")
    t_obs = np.array([float(r["time_hr"]) for r in obs_pressure_rows], float)
    dp_obs = np.array([float(r["dp_psi"]) for r in obs_pressure_rows], float)

    source = SourceWell(
        ar["well"], float(ar["x_ft"]), float(ar["y_ft"]),
        rate_levels_to_events(t_rate, q_level), active_rw,
    )
    observation = ObservationWell(
        orow["well"], float(orow["x_ft"]), float(orow["y_ft"]), obs_rw,
    )
    reservoir_template = Reservoir(
        h_ft=_f(inputs, "h_ft"), mu_cp=_f(inputs, "mu_cp"),
        B_rb_stb=_f(inputs, "B_rb_stb", 1.0),
    )
    return inputs, model, t_obs, dp_obs, q_level, source, observation, reservoir_template


def _nominal_rate(q_level: np.ndarray) -> float:
    nz = q_level[np.abs(q_level) > 0]
    if nz.size == 0:
        raise ValueError("rates.csv contains no non-zero rate level.")
    return float(nz[0])


def _distance_ft(source: SourceWell, observation: ObservationWell) -> float:
    return float(np.hypot(observation.x_ft - source.x_ft, observation.y_ft - source.y_ft))


def _type_curve_limits(u_data: np.ndarray) -> np.ndarray:
    positive = u_data[np.isfinite(u_data) & (u_data > 0)]
    if positive.size == 0:
        return np.logspace(-3, 3, 500)
    lo = max(float(np.min(positive)) / 3.0, 1e-5)
    hi = max(float(np.max(positive)) * 2.0, lo * 100.0)
    return np.logspace(np.log10(lo), np.log10(hi), 500)


def _make_plot(case_name, inputs, t_obs, dp_obs, q_level, k_md, pct, prediction_fn, subtitle):
    res_fit = Reservoir(
        h_ft=_f(inputs, "h_ft"), mu_cp=_f(inputs, "mu_cp"),
        B_rb_stb=_f(inputs, "B_rb_stb", 1.0), phi_ct_psi_inv=pct,
    )
    source, observation = prediction_fn.__dict__["source"], prediction_fn.__dict__["observation"]
    q_ref = _nominal_rate(q_level)
    r_ft = _distance_ft(source, observation)
    u_data = dimensionless_time_over_distance_squared(t_obs, k_md, res_fit, r_ft)
    pD_data = dimensionless_pressure_constant_rate(dp_obs, q_ref, k_md, res_fit)
    u_curve = _type_curve_limits(u_data)
    t_curve = u_curve * res_fit.mu_cp * res_fit.phi_ct * r_ft**2 / (0.0002637 * k_md)
    pD_curve = dimensionless_pressure_constant_rate(prediction_fn(t_curve), q_ref, k_md, res_fit)

    positive_t = t_obs[t_obs > 0]
    tmin = max(0.001, float(np.min(positive_t))/20.0) if positive_t.size else 0.001
    t_dense = np.linspace(tmin, float(np.max(t_obs))*1.08, 700)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(t_obs, dp_obs, label="observed")
    axes[0].plot(t_dense, prediction_fn(t_dense), label="matched model")
    axes[0].set_xlabel("time, h")
    axes[0].set_ylabel(r"signed pressure drop $p_i-p$, psi")
    axes[0].set_title(f"{_s(inputs, 'case_name', case_name)} — pressure history")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    add_type_curve_match(
        axes[1], u_curve, pD_curve, u_data, pD_data,
        curve_label="matched model type curve", data_label="scaled test data",
        reference_u=u_curve, reference_pD=theis_well_function(u_curve),
        reference_label="infinite-acting Theis",
    )
    axes[1].set_title(f"{_s(inputs, 'case_name', case_name)} — type-curve match")
    fig.suptitle(subtitle)
    fig.tight_layout()
    fig_path = RESULTS / f"{case_name}.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _attach_wells(fn, source, observation):
    # Simple metadata used by the shared plotting helper without creating another class.
    fn.__dict__["source"] = source
    fn.__dict__["observation"] = observation
    return fn


def run_case(case_name: str):
    inputs, model, t_obs, dp_obs, q_level, source, observation, template = _load_case(case_name)
    loss = _s(inputs, "robust_loss", "soft_l1")
    lines = [_s(inputs, "case_name", case_name), f"model = {model}"]

    if model == "line_source":
        fit = fit_line_source_auto(t_obs, dp_obs, source, observation, template, robust_loss=loss)
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=fit.phi_ct_psi_inv)
        pred = _attach_wells(lambda t: simulate_line_source(t, source, observation, fit.k_md, res), source, observation)
        k, pct, rmse = fit.k_md, fit.phi_ct_psi_inv, fit.rmse
        lines += [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi"]
        subtitle = rf"Matched: $k={k:.3f}$ mD, $\phi c_t={pct:.3e}$ psi$^{{-1}}$, RMSE={rmse:.2f} psi"

    elif model == "closed_rectangle":
        aspect = _f(inputs, "aspect_ratio_width_over_height")
        sxD = _f(inputs, "source_xD"); syD = _f(inputs, "source_yD")
        angle = _f(inputs, "rectangle_angle_deg", 0.0)
        fit = fit_closed_rectangle_auto(
            t_obs, dp_obs, source, observation, template, aspect, sxD, syD,
            rectangle_angle_deg=angle, robust_loss=loss,
        )
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=fit.phi_ct_psi_inv)
        rect = ClosedRectangle(0.0, 0.0, fit.width_ft, fit.height_ft)
        src_local = SourceWell(source.name, sxD*fit.width_ft, syD*fit.height_ft, source.events, source.rw_ft)
        obs_local = ObservationWell(
            observation.name, fit.observation_xD*fit.width_ft,
            fit.observation_yD*fit.height_ft, observation.rw_ft,
        )
        pred = _attach_wells(
            lambda t: simulate_closed_rectangle(t, src_local, obs_local, fit.k_md, res, rect),
            source, observation,
        )
        k, pct, rmse = fit.k_md, fit.phi_ct_psi_inv, fit.rmse
        lines += [
            f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi",
            f"width = {fit.width_ft:.6f} ft", f"height = {fit.height_ft:.6f} ft",
            f"area = {fit.area_acres:.6f} acres",
            f"observation xD = {fit.observation_xD:.6f}",
            f"observation yD = {fit.observation_yD:.6f}",
        ]
        subtitle = rf"Matched: $k={k:.2f}$ mD, $\phi c_t={pct:.3e}$ psi$^{{-1}}$, A={fit.area_acres:.1f} acres"

    elif model == "leaky":
        fit = fit_leaky_auto(t_obs, dp_obs, source, observation, template, robust_loss=loss)
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=fit.phi_ct_psi_inv)
        pred = _attach_wells(
            lambda t: simulate_leaky_line_source(t, source, observation, fit.k_md, res, fit.leakage_factor_ft),
            source, observation,
        )
        k, pct, rmse = fit.k_md, fit.phi_ct_psi_inv, fit.rmse
        lines += [f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi", f"leakage factor B = {fit.leakage_factor_ft:.6f} ft"]
        subtitle = rf"Matched: $k={k:.3f}$ mD, $\phi c_t={pct:.3e}$ psi$^{{-1}}$, B={fit.leakage_factor_ft:.1f} ft"

    elif model == "active_storage_skin":
        fit = fit_active_storage_skin_auto(t_obs, dp_obs, source, observation, template, robust_loss=loss)
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=fit.phi_ct_psi_inv)
        pred = _attach_wells(
            lambda t: simulate_active_storage_skin(t, source, observation, fit.k_md, res, fit.CD_active, fit.skin_active),
            source, observation,
        )
        k, pct, rmse = fit.k_md, fit.phi_ct_psi_inv, fit.rmse
        lines += [
            f"k = {k:.6f} mD", f"phi*ct = {pct:.8e} 1/psi",
            f"CD active = {fit.CD_active:.8g}", f"skin active = {fit.skin_active:.6f}",
            f"C active = {fit.C_active_bbl_per_psi:.8g} bbl/psi",
        ]
        subtitle = rf"Matched: $k={k:.3f}$ mD, $\phi c_t={pct:.3e}$ psi$^{{-1}}$, $C_D={fit.CD_active:.2g}$, S={fit.skin_active:.2f}"

    elif model == "double_porosity":
        inter = _s(inputs, "interporosity_model").lower()
        fit = fit_double_porosity_auto(
            t_obs, dp_obs, source, observation, template,
            interporosity=inter, robust_loss=loss,
        )
        res = Reservoir(template.h_ft, template.mu_cp, template.B_rb_stb, phi_ct_psi_inv=fit.phi_ct_total_psi_inv)
        pred = _attach_wells(
            lambda t: simulate_double_porosity(
                t, source, observation, fit.k_fracture_md, res, float(source.rw_ft),
                fit.omega, fit.lam, interporosity=fit.interporosity,
            ), source, observation,
        )
        k, pct, rmse = fit.k_fracture_md, fit.phi_ct_total_psi_inv, fit.rmse
        lines += [
            f"k fracture = {k:.6f} mD", f"phi*ct total = {pct:.8e} 1/psi",
            f"omega = {fit.omega:.8g}", f"lambda = {fit.lam:.8g}",
            f"interporosity = {fit.interporosity}",
        ]
        subtitle = rf"Matched: $k_f={k:.3f}$ mD, $\phi c_t={pct:.3e}$ psi$^{{-1}}$, $\omega={fit.omega:.3g}$, $\lambda={fit.lam:.3g}$"

    else:  # validate_input_schema should make this unreachable
        raise ValueError(f"Unsupported CSV-runner model: {model}")

    ct = _optional_f(inputs.get("ct_psi_inv", ""))
    if ct is not None:
        lines.append(f"phi = {pct/ct:.6f}")
    lines.append(f"RMSE = {rmse:.6f} psi")

    fig_path = _make_plot(case_name, inputs, t_obs, dp_obs, q_level, k, pct, pred, subtitle)
    text = "\n".join(lines) + "\n"
    result_path = RESULTS / f"{case_name}.txt"
    result_path.write_text(text, encoding="utf-8")
    return fig_path, result_path, text


def available_cases() -> list[str]:
    return sorted(
        p.name for p in EXAMPLES.iterdir()
        if p.is_dir() and all((p / f).exists() for f in ("inputs.csv", "wells.csv", "rates.csv", "pressure.csv"))
    )


def main():
    parser = argparse.ArgumentParser(description="Run a CSV-defined interference-test case.")
    parser.add_argument("case", nargs="?", help="case folder name under examples/")
    parser.add_argument("--all", action="store_true", help="run all CSV example cases")
    parser.add_argument("--list", action="store_true", help="list available cases")
    args = parser.parse_args()

    cases = available_cases()
    if args.list:
        print("\n".join(cases)); return
    selected = cases if args.all else ([args.case] if args.case else None)
    if selected is None:
        parser.error("provide a case name, --all, or --list")

    for case in selected:
        fig_path, result_path, text = run_case(case)
        print(f"\n[{case}]\n{text.rstrip()}\nfigure: {fig_path}\nresults: {result_path}")


if __name__ == "__main__":
    main()
