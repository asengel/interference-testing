"""User-facing case input names, legacy aliases and supported model metadata."""
from __future__ import annotations
from dataclasses import dataclass

# Only relevant rows are needed; inputs.csv may also have a `description` column.
INPUT_SCHEMA = (
    "case_name", "model", "run_mode", "permeability_md", "storativity_psi_inv",
    "rectangle_height_ft", "leakage_factor_ft", "active_well_storage_cd", "active_well_skin",
    "fracture_storage_fraction", "interporosity_flow_parameter",
    "simulation_start_hr", "simulation_end_hr", "simulation_points", "thickness_ft", "viscosity_cp", "formation_volume_factor_rb_stb",
    "total_compressibility_psi_inv", "fit_loss",
    "boundary_type", "boundary_distance_ft", "boundary_azimuth_deg",
    "rectangle_width_height_ratio", "active_well_x_percent", "active_well_y_percent",
    "rectangle_rotation_deg", "interporosity_model",
    # Former parameter names remain valid for existing case folders.
    "h_ft", "mu_cp", "B_rb_stb", "ct_psi_inv", "robust_loss",
    "aspect_ratio_width_over_height", "source_xD", "source_yD", "rectangle_angle_deg",
)
WELL_COLUMNS = ("well", "role", "x_ft", "y_ft", "rw_ft")
RATE_COLUMNS = ("well", "time_hr", "rate_stbd")
PRESSURE_COLUMNS = ("well", "time_hr", "dp_psi")

ALIASES = {
    "thickness_ft": "h_ft",
    "viscosity_cp": "mu_cp",
    "formation_volume_factor_rb_stb": "B_rb_stb",
    "total_compressibility_psi_inv": "ct_psi_inv",
    "fit_loss": "robust_loss",
    "rectangle_width_height_ratio": "aspect_ratio_width_over_height",
    "active_well_x_percent": "source_xD",
    "active_well_y_percent": "source_yD",
    "rectangle_rotation_deg": "rectangle_angle_deg",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    required_inputs: tuple[str, ...]
    require_active_rw: bool = False
    require_observation_rw: bool = False
    fitted_parameters: tuple[str, ...] = ()
    notes: str = ""


MODEL_SPECS = {
    "line_source": ModelSpec("line_source", ("h_ft", "mu_cp", "B_rb_stb"),
        fitted_parameters=("k_md", "phi_ct_psi_inv"), notes="Infinite-acting line-source or one straight boundary."),
    "closed_rectangle": ModelSpec("closed_rectangle",
        ("h_ft", "mu_cp", "B_rb_stb", "aspect_ratio_width_over_height", "source_xD", "source_yD"),
        fitted_parameters=("k_md", "phi_ct_psi_inv", "rectangle_size"),
        notes="Closed no-flow rectangle; all four edges represented by image wells."),
    "leaky": ModelSpec("leaky", ("h_ft", "mu_cp", "B_rb_stb"),
        fitted_parameters=("k_md", "phi_ct_psi_inv", "leakage_factor_ft")),
    "active_storage_skin": ModelSpec("active_storage_skin", ("h_ft", "mu_cp", "B_rb_stb"),
        require_active_rw=True,
        fitted_parameters=("k_md", "phi_ct_psi_inv", "CD_active", "skin_active")),
    "double_porosity": ModelSpec("double_porosity",
        ("h_ft", "mu_cp", "B_rb_stb", "interporosity_model"), require_active_rw=True,
        fitted_parameters=("k_fracture_md", "phi_ct_total_psi_inv", "omega", "lambda")),
}


def normalize_model_name(name: str) -> str:
    n = str(name).strip().lower()
    return {
        "theis": "line_source", "infinite": "line_source", "hantush": "leaky",
        "hantush_jacob": "leaky", "storage_skin": "active_storage_skin",
        "storage_skin_active": "active_storage_skin",
        "double_porosity_pss": "double_porosity",
        "double_porosity_transient_slab": "double_porosity",
        "double_porosity_transient_sphere": "double_porosity",
    }.get(n, n)


def canonicalize_inputs(values: dict[str, str]) -> dict[str, str]:
    extra = sorted(set(values) - set(INPUT_SCHEMA))
    if extra:
        raise ValueError(f"Unknown inputs.csv parameter(s): {', '.join(extra)}. Check spelling.")
    out: dict[str, str] = {}
    for key, val in values.items():
        value = str(val).strip()
        if key in {"active_well_x_percent", "active_well_y_percent"} and value:
            try:
                fraction = float(value)/100.0
            except ValueError as exc:
                raise ValueError(f"{key} must be a percentage between 0 and 100.") from exc
            value = str(fraction)
        canonical = ALIASES.get(key, key)
        if canonical in out and value and out[canonical] and value != out[canonical]:
            raise ValueError(f"Conflicting input values for {canonical}; use one name only.")
        if value or canonical not in out:
            out[canonical] = value
    return out


def validate_input_schema(values: dict[str, str]) -> None:
    from .validation import positive_number, finite_number
    v = canonicalize_inputs(values)
    model = normalize_model_name(v.get("model", ""))
    mode = v.get("run_mode", "auto").lower() or "auto"
    if mode not in {"manual", "auto"}:
        raise ValueError("run_mode must be 'manual' or 'auto'.")
    if model not in MODEL_SPECS:
        raise ValueError(f"Unsupported model {v.get('model')!r}. Choices: {', '.join(MODEL_SPECS)}.")
    for required in MODEL_SPECS[model].required_inputs:
        if not v.get(required, ""):
            raise ValueError(f"Model '{model}' requires '{required}' in inputs.csv.")
    for name in ("h_ft", "mu_cp", "B_rb_stb"):
        positive_number(v[name], name)
    if v.get("ct_psi_inv"):
        positive_number(v["ct_psi_inv"], "ct_psi_inv")
    if v.get("robust_loss", "linear") not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
        raise ValueError("fit_loss must be linear, soft_l1, huber, cauchy or arctan.")
    if model == "closed_rectangle":
        positive_number(v["aspect_ratio_width_over_height"], "rectangle width/height ratio")
        for key in ("source_xD", "source_yD"):
            z = finite_number(v[key], key)
            if not (0 < z < 1):
                raise ValueError(f"{key} must place the active well strictly inside the rectangle (0-100%).")
        if v.get("rectangle_angle_deg"):
            finite_number(v["rectangle_angle_deg"], "rectangle angle")
    if model == "double_porosity" and v.get("interporosity_model") not in {"pss", "transient_slab", "transient_sphere"}:
        raise ValueError("interporosity_model must be pss, transient_slab or transient_sphere.")
    if mode == "manual":
        from .validation import (K_BOUNDS_MD, STORATIVITY_BOUNDS_PER_PSI,
                                 LEAKAGE_BOUNDS_FT, LAMBDA_BOUNDS)
        def number_in_range(key, low, high, *, strict_low=False, strict_high=False):
            if not v.get(key):
                raise ValueError(f"manual mode requires '{key}' in inputs.csv.")
            x = finite_number(v[key], key)
            if (x <= low if strict_low else x < low) or (x >= high if strict_high else x > high):
                raise ValueError(f"{key} must be within {('(' if strict_low else '[')}{low:g}, {high:g}{(')' if strict_high else ']')}.")
            return x
        k = number_in_range("permeability_md", *K_BOUNDS_MD)
        pct = number_in_range("storativity_psi_inv", *STORATIVITY_BOUNDS_PER_PSI)
        if v.get("ct_psi_inv") and pct > positive_number(v["ct_psi_inv"], "total_compressibility_psi_inv"):
            raise ValueError("storativity_psi_inv / total_compressibility_psi_inv cannot exceed 1 (porosity).")
        if model == "closed_rectangle":
            number_in_range("rectangle_height_ft", 1e-3, 1e9)
        if model == "leaky":
            number_in_range("leakage_factor_ft", *LEAKAGE_BOUNDS_FT)
        if model == "active_storage_skin":
            number_in_range("active_well_storage_cd", 1e-12, 1e14)
            number_in_range("active_well_skin", 0, 50)
        if model == "double_porosity":
            number_in_range("fracture_storage_fraction", 0, 1, strict_low=True, strict_high=True)
            number_in_range("interporosity_flow_parameter", *LAMBDA_BOUNDS)
    for key in ("simulation_start_hr", "simulation_end_hr"):
        if v.get(key):
            x = finite_number(v[key], key)
            if x < 0:
                raise ValueError(f"{key} must be nonnegative.")
    if v.get("simulation_start_hr") and v.get("simulation_end_hr"):
        if float(v["simulation_end_hr"]) <= float(v["simulation_start_hr"]):
            raise ValueError("simulation_end_hr must exceed simulation_start_hr.")
    if v.get("simulation_points"):
        n = finite_number(v["simulation_points"], "simulation_points")
        if int(n) != n or not (2 <= n <= 20000):
            raise ValueError("simulation_points must be an integer between 2 and 20000.")
    kind = v.get("boundary_type", "none").strip().lower() or "none"
    if kind not in {"none", "no_flow", "constant_pressure"}:
        raise ValueError("boundary_type must be none, no_flow or constant_pressure.")
    if kind != "none":
        if model != "line_source":
            raise ValueError("A single straight boundary is supported only with model=line_source. The closed_rectangle model has its own four no-flow edges.")
        # Manual mode fixes the geometric distance. Automatic mode estimates it;
        # a supplied value is merely an optional starting guess for optimization.
        if mode == "manual":
            positive_number(v.get("boundary_distance_ft"), "boundary_distance_ft")
        elif v.get("boundary_distance_ft"):
            positive_number(v["boundary_distance_ft"], "initial boundary_distance_ft")
        if not v.get("boundary_azimuth_deg"):
            raise ValueError("boundary_azimuth_deg is required for a boundary (use 0 for +X direction).")
        finite_number(v["boundary_azimuth_deg"], "boundary_azimuth_deg")
    elif v.get("boundary_distance_ft") or v.get("boundary_azimuth_deg"):
        raise ValueError("Boundary distance/azimuth were provided but boundary_type is none.")


def validate_well_radius(model: str, active_rw_ft: float | None, observation_rw_ft: float | None) -> None:
    from .validation import positive_number
    spec = MODEL_SPECS[normalize_model_name(model)]
    if spec.require_active_rw:
        positive_number(active_rw_ft, "Active well radius rw_ft")
    if spec.require_observation_rw:
        positive_number(observation_rw_ft, "Observation well radius rw_ft")
