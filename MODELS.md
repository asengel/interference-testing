# Model map

The CSV runner uses one common input schema. **Known quantities are inputs; parameters intended to be inferred from pressure data are fitted.**

| Model name | Purpose | 
|---|---|---|---|---|
| `line_source` | Infinite homogeneous reservoir | 
| `closed_rectangle` | Closed bounded rectangle | 
| `leaky` | Hantush-Jacob leakage | 
| `active_storage_skin` | Storage/skin at active well | 
| `double_porosity` | PSS or transient interporosity transfer | 

The underlying library also contains linear image-boundary models, homogeneous anisotropy, multiwell superposition, and a two-well storage/skin kernel. These are not yet exposed through the single-pair CSV runner because their user-facing parameterization still needs the same level of validation as the models above.

## Important design rule

Do not add an inferred model parameter to `inputs.csv` merely because an optimizer needs a starting value. Starting points belong inside the solver. A model-specific physical constraint belongs in `inputs.csv` or `wells.csv` only when it is genuinely known before interpretation.
