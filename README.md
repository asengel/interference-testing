# Interference Testing — Automated history matching

The user supplies **known physical data only**. Parameters that are supposed to be inferred from the interference test are fitted automatically.
The user selects the physical model and supplies the measured pressure history, rate history, well geometry, and known reservoir/fluid properties. The program then estimates the unknown model parameters by nonlinear least-squares history matching.

## Project layout

```text
interference_testing/
├── interference/                 # analytical models + matching engine
├── examples/
│   ├── run.py                    # one generic runner
│   ├── earlougher_9_1/
│   │   ├── inputs.csv            # reservoir/fluid settings + model selection
│   │   ├── wells.csv             # well coordinates + optional rw
│   │   ├── rates.csv             # rate histories by well
│   │   └── pressure.csv          # observed pressure by well
│   └── earlougher_9_2/
│       └── same four CSV files
├── results/                      # generated figures + matched parameters
├── MODELS.md
├── README.md
└── pyproject.toml
```

## Run a case

From the project directory:

```bash
python examples/run.py --list
python examples/run.py earlougher_9_1
python examples/run.py earlougher_9_2
python examples/run.py --all
```

Each run creates:

- a pressure-history match;
- a dimensionless type-curve match;
- a text file containing the fitted parameters.

## The four input files

### `inputs.csv`

`inputs.csv` contains reservoir/fluid information and the selected analytical model. It deliberately does **not** contain well coordinates or well radius.

Every case uses exactly the same row schema:

```csv
parameter,value
case_name,My interference test
model,line_source
h_ft,100
mu_cp,0.5
B_rb_stb,1.0
ct_psi_inv,
robust_loss,soft_l1
aspect_ratio_width_over_height,
source_xD,
source_yD,
rectangle_angle_deg,
interporosity_model,
```

Rows not required by the selected model remain blank.

### `wells.csv`

All cases use the same columns:

```csv
well,role,x_ft,y_ft,rw_ft
A,active,0,0,
B,observation,3773,0,
```

`rw_ft` is optional unless the selected model needs finite-wellbore physics.

For example, the following is valid for an active-well storage/skin model:

```csv
well,role,x_ft,y_ft,rw_ft
A,active,0,0,0.2917
B,observation,3773,0,0.2917
```

The current runner uses one active and one observation well per case. The well-name column is retained so the CSV format can extend to multiple active/observation wells without redesigning the files.

### `rates.csv`

```csv
well,time_hr,rate_stbd
A,0,-170
A,48,0
```

These are rate **levels**, not rate-change increments. The program converts them to superposition events internally.

Sign convention:

```text
production  > 0
injection   < 0
```

### `pressure.csv`

```csv
well,time_hr,dp_psi
B,4.3,-22
B,21.6,-82
```

Pressure convention:

```text
dp = p_initial - p_observed
```

Therefore production normally produces positive `dp`, while injection normally produces negative `dp`.

## Model-specific requirements

The same CSV columns/rows are used for every model. The selected model controls what is required and what is ignored.

| Quantity | line_source | closed_rectangle | leaky | active_storage_skin | double_porosity |
|---|---:|---:|---:|---:|---:|
| `h_ft` | required | required | required | required | required |
| `mu_cp` | required | required | required | required | required |
| `B_rb_stb` | required | required | required | required | required |
| well coordinates | required | required | required | required | required |
| active-well `rw_ft` | ignored | ignored | ignored | **required** | **required** |
| observation `rw_ft` | ignored | ignored | ignored | ignored | ignored |
| rectangle aspect/source position | ignored | **required** | ignored | ignored | ignored |
| `interporosity_model` | ignored | ignored | ignored | ignored | **required** |
| permeability | **fitted** | **fitted** | **fitted** | **fitted** | **fitted** |
| `phi*ct` | **fitted** | **fitted** | **fitted** | **fitted** | **fitted** |
| rectangle size | — | **fitted** | — | — | — |
| leakage factor | — | — | **fitted** | — | — |
| active well storage | — | — | — | **fitted** | — |
| active skin | — | — | — | **fitted** | — |
| `omega`, `lambda` | — | — | — | — | **fitted** |

`ct_psi_inv` is optional for all models. It is not used to fit `phi*ct`. If supplied, the result file also reports:

```text
phi = (phi*ct) / ct
```

## Current runner models

### `line_source`

Fits:

```text
k
phi*ct
```

### `closed_rectangle`

Requires the known geometric constraints:

```text
aspect_ratio_width_over_height
source_xD
source_yD
```

`rectangle_angle_deg` is optional and defaults to zero.

Fits:

```text
k
phi*ct
rectangle width
rectangle height
rectangle area
```

### `leaky`

Hantush-Jacob model. Fits:

```text
k
phi*ct
leakage factor B
```

### `active_storage_skin`

Requires active-well `rw_ft` in `wells.csv`.

Fits:

```text
k
phi*ct
CD_active
skin_active
C_active (bbl/psi)
```

The current Stehfest implementation deliberately restricts the fitted skin to nonnegative values because the transformed infinitesimal-skin solution can develop a positive-real-axis pole for negative skin.

### `double_porosity`

Requires active-well `rw_ft` and one of:

```text
interporosity_model,pss
interporosity_model,transient_slab
interporosity_model,transient_sphere
```

Fits:

```text
k_fracture
phi*ct_total
omega
lambda
```

## Type-curve figures

Every runner model produces a two-panel figure:

1. observed pressure and best-fit physical pressure history;
2. scaled pressure data and the corresponding matched dimensionless type curve, with the infinite-acting Theis curve shown as a reference.
