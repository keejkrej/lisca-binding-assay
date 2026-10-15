# Python and Rust Parity (`lisca-binding-assay`)

`lisca-binding-assay` provides both a Python CLI (`binding`) and an on-par Rust crate/CLI (`lisca-binding` / `lisca-analyze`) for single-particle detection and binding assay analysis.

## Architecture

- **Python implementation**:
  - Uses `picassosr==0.11.3` (`picasso.localize`, `picasso.gausslq`, `picasso.gaussmle`, `picasso.avgroi`) in `src/binding/core/picasso.py` and `src/binding/services/picasso.py`.
  - CLI command: `binding picasso` (supporting both ROI workspace mode `--pos` and raw TIFF stack mode `--input`/`--output`).
- **Rust implementation**:
  - Uses `picasso` (`/home/jack/workspace/picasso-rs`), `marray`, and `mplot` in `crates/lisca-binding`.
  - CLI binary: `lisca-analyze` with subcommands `picasso`, `filter-spots`, `spot-counts`, `traces`, `show-traces`, `plot-lnp`, and `pipeline`.

## Shared 15-Column Spot CSV Schema

Both Python and Rust emit identical 15-column CSV files (`spots_t{frame:04d}.csv` in ROI mode, `t{frame:04d}_c{channel:03d}_spots.csv` in stack mode):

```text
y,x,intensity,probability,fwhm,intens_A,intens_B,snb,photons,sx,sy,lpx,lpy,ellipticity,net_gradient
```

## Verification

- **Python unit tests**:
  ```bash
  .venv/bin/pytest
  ```
- **Rust unit & Python-vs-Rust parity tests**:
  ```bash
  cargo test --test binding_parity
  ```
