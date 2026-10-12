# Models and Required Inputs

Each case uses `inputs.csv`, `wells.csv`, and `rates.csv`. The `pressure.csv` file is required for automatic fitting and optional for manual simulation.

Set `run_mode=manual` to simulate using specified parameters, or `run_mode=auto` (default) to estimate parameters from measured pressures. All models require `thickness_ft`, `viscosity_cp`, and `formation_volume_factor_rb_stb`. Manual mode additionally requires `permeability_md` and `storativity_psi_inv` (φct).

## Model Summary

| Model | Description | Additional inputs | Manual parameters / Auto fitting |
|---|---|---|---|
| `line_source` | Infinite-acting Theis solution; optional no-flow or constant-pressure boundary | Optional `boundary_type`, `boundary_azimuth_deg`, `boundary_distance_ft` | Manual: k, φct, fixed boundary distance if applicable. Auto: fits k, φct, and boundary distance if applicable |
| `closed_rectangle` | Closed rectangular reservoir with four impermeable boundaries | `rectangle_width_height_ratio`, `active_well_x_percent`, `active_well_y_percent`; optional `rectangle_rotation_deg` | Manual: `rectangle_height_ft`. Auto: fits rectangle dimensions |
| `leaky` | Hantush–Jacob leaky-reservoir solution | None | Manual: `leakage_factor_ft`. Auto: fits leakage factor |
| `active_storage_skin` | Active-well storage and skin effects | Active-well `rw_ft` in `wells.csv` | Manual: `active_well_storage_cd`, `active_well_skin`. Auto: fits storage and skin; skin ≥ 0 |
| `double_porosity` | Pseudo-steady or transient interporosity flow | `interporosity_model`: `pss`, `transient_slab`, or `transient_sphere`; active-well `rw_ft` | Manual: `fracture_storage_fraction` (0 < ω < 1), `interporosity_flow_parameter` (λ > 0). Auto: fits ω and λ |

All manual models require permeability and storativity in addition to the model-specific parameters listed above.

## Boundary Geometry

For `line_source`, one straight boundary can be specified using `boundary_type=no_flow` or `constant_pressure`. The `boundary_distance_ft` is measured perpendicularly from the active well, while `boundary_azimuth_deg` defines the direction towards the boundary, counterclockwise from +X. Both wells must lie on the same side of the boundary.

In `auto` mode, boundary distance is fitted jointly with permeability and storativity; a supplied distance is used only as an initial guess. Boundary azimuth remains fixed. In `manual` mode, boundary distance must be specified.

For `closed_rectangle`, well locations are defined as percentages of rectangle width and height, measured from the left and bottom edges.

## Simulation and Outputs

Use `simulation_start_hr` (default 0), `simulation_end_hr`, and `simulation_points` (default 700) to control simulation times. `simulation_end_hr` is required when no pressure observations are supplied.

Results include fitted or specified parameters, RMSE when observations are available, a four-panel diagnostic figure (pressure response, dimensionless type curve, residuals, and rate history), and `<case>_predicted.csv`.

**Notes:** The CSV workflow supports one active and one observation well. Only include parameters relevant to the selected model. Optional `total_compressibility_psi_inv` enables porosity inference from φct. Fitted parameters are bounded for numerical stability, but a good pressure match does not guarantee parameter uniqueness.