# Model map

The CSV runner uses one common input schema. **Known quantities are inputs; parameters intended to be inferred from pressure data are fitted.**

| Model name | Purpose | 
|---|---|---|---|---|
| `line_source` | Infinite homogeneous reservoir | 
| `closed_rectangle` | Closed bounded rectangle | 
| `leaky` | Hantush-Jacob leakage | 
| `active_storage_skin` | Storage/skin at active well | 
| `double_porosity` | PSS or transient interporosity transfer | 

The underlying library also contains linear image-boundary models, homogeneous anisotropy, multiwell superposition, and a two-well storage/skin kernel.
