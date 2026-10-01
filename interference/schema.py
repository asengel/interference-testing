from __future__ import annotations

from dataclasses import dataclass

INPUT_SCHEMA = (
    "case_name",
    "model",
    "h_ft",
    "mu_cp",
    "B_rb_stb",
    "ct_psi_inv",
    "robust_loss",
    # closed rectangle
    "aspect_ratio_width_over_height",
    "source_xD",
    "source_yD",
    "rectangle_angle_deg",
    # double porosity
    "interporosity_model",
)

WELL_COLUMNS = ("well", "role", "x_ft", "y_ft", "rw_ft")
RATE_COLUMNS = ("well", "time_hr", "rate_stbd")
PRESSURE_COLUMNS = ("well", "time_hr", "dp_psi")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    required_inputs: tuple[str, ...]
    require_active_rw: bool = False
    require_observation_rw: bool = False
    fitted_parameters: tuple[str, ...] = ()
    notes: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "line_source": ModelSpec(
        name="line_source",
        required_inputs=("h_ft", "mu_cp", "B_rb_stb"),
        fitted_parameters=("k_md", "phi_ct_psi_inv"),
        notes="Infinite-acting line-source/Theis interference model.",
    ),
    "closed_rectangle": ModelSpec(
        name="closed_rectangle",
        required_inputs=(
            "h_ft", "mu_cp", "B_rb_stb",
            "aspect_ratio_width_over_height", "source_xD", "source_yD",
        ),
        fitted_parameters=("k_md", "phi_ct_psi_inv", "rectangle_size"),
        notes="Closed rectangular reservoir with no-flow outer boundaries.",
    ),
    "leaky": ModelSpec(
        name="leaky",
        required_inputs=("h_ft", "mu_cp", "B_rb_stb"),
        fitted_parameters=("k_md", "phi_ct_psi_inv", "leakage_factor_ft"),
        notes="Hantush-Jacob leaky-reservoir interference model.",
    ),
    "active_storage_skin": ModelSpec(
        name="active_storage_skin",
        required_inputs=("h_ft", "mu_cp", "B_rb_stb"),
        require_active_rw=True,
        fitted_parameters=(
            "k_md", "phi_ct_psi_inv", "CD_active", "skin_active", "C_active_bbl_per_psi"
        ),
        notes="Finite-radius active-well storage/skin model. Negative skin is intentionally not fitted.",
    ),
    "double_porosity": ModelSpec(
        name="double_porosity",
        required_inputs=("h_ft", "mu_cp", "B_rb_stb", "interporosity_model"),
        require_active_rw=True,
        fitted_parameters=("k_fracture_md", "phi_ct_total_psi_inv", "omega", "lambda"),
        notes="Double-porosity interference model; interporosity_model selects pss/transient_slab/transient_sphere.",
    ),
}


def normalize_model_name(name: str) -> str:
    n = name.strip().lower()
    aliases = {
        "theis": "line_source",
        "infinite": "line_source",
        "hantush": "leaky",
        "hantush_jacob": "leaky",
        "storage_skin": "active_storage_skin",
        "storage_skin_active": "active_storage_skin",
        "double_porosity_pss": "double_porosity",
        "double_porosity_transient_slab": "double_porosity",
        "double_porosity_transient_sphere": "double_porosity",
    }
    return aliases.get(n, n)


def validate_input_schema(values: dict[str, str]) -> None:
    missing_rows = [k for k in INPUT_SCHEMA if k not in values]
    if missing_rows:
        raise ValueError(
            "inputs.csv does not use the standard schema. Missing rows: "
            + ", ".join(missing_rows)
        )

    model = normalize_model_name(values.get("model", ""))
    if model not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported model {values.get('model')!r}. Supported models: "
            + ", ".join(sorted(MODEL_SPECS))
        )
    for key in MODEL_SPECS[model].required_inputs:
        if not values.get(key, "").strip():
            raise ValueError(f"Selected model '{model}' requires input '{key}'.")


def validate_well_radius(model: str, active_rw_ft: float | None, observation_rw_ft: float | None) -> None:
    spec = MODEL_SPECS[normalize_model_name(model)]
    if spec.require_active_rw and (active_rw_ft is None or active_rw_ft <= 0):
        raise ValueError(
            f"Selected model '{spec.name}' requires rw_ft for the active well in wells.csv."
        )
    if spec.require_observation_rw and (observation_rw_ft is None or observation_rw_ft <= 0):
        raise ValueError(
            f"Selected model '{spec.name}' requires rw_ft for the observation well in wells.csv."
        )
