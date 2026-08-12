# binding (`lisca-binding-assay`)

CLI tools and analysis scripts for **2D LNP membrane-binding** on LISCA ROI crops:
Spotiflow detection → filter → per-cell counts → figures, plus reaction–diffusion
theory and presentation asset exporters.

> **3D volumetric particle / membrane workflows** (watershed, `analyze-membrane`,
> multi-z Spotiflow seeds, …) were removed from `main`. They remain on branch
> [`3d`](https://github.com/keejkrej/lisca-binding-assay/tree/3d) if needed.

## Install

```bash
uv sync
```

The `binding` entry point is available in the project virtual environment
(`.venv/bin/binding`).

## Input layout

Most commands expect data converted from ND2 (or similar) into this folder structure:

```text
dataset/
  Pos0/
    img_channel000_position000_time000000000_z000.tif
    ...
  roi/
    Pos0/
      index.json
      Roi0.tif
      ...
  bbox/
    Pos0.csv
```

Filename pattern: `img_channel{C}_position{P}_time{T}_z{Z}.tif`

## Workflows

### 1. LNP spot counting per cell (2D Spotiflow) — primary

```text
spotiflow --roi-stacks -> filter-spots -> spot-counts -> plot-lnp
```

```bash
DATA=~/data/lisca_review/fig5/20260324_1

binding spotiflow "$DATA" \
  --roi-stacks --all-rois --all-times \
  -p 0 -c 1 \
  --model general --estimate-params \
  -o "$DATA/results/spotiflow"

binding filter-spots "$DATA/results/spotiflow" -o "$DATA/results/filtered" \
  --min-intensity 4000 \
  --min-fwhm 2.0 --max-fwhm 6.0 \
  --min-probability 0.4

binding spot-counts "$DATA/results/filtered" -o "$DATA/results/" \
  --time-interval 40 --cumulative

binding plot-lnp "$DATA" "$DATA/results/spot_counts_position000_channel001.csv" \
  --filtered-dir "$DATA/results/filtered" \
  -o "$DATA/results/fig5_panels.png" \
  -c 1 --roi 2 --time 72 --time-unit min
```

**Time axis:** `spot-counts` stores `time_real` in seconds. `plot-lnp` displays
minutes by default. For subsampled ND2 where every 10th frame was kept from a
4 s acquisition, use `--time-interval 40`.

### 2. ROI mean-intensity time course

Fixed square ROIs over time (not spot counting):

```bash
binding timeseries DATA -p 0 -c 1 \
  --sizes 8,16,32,64,128 \
  --center-y 128 --center-x 128 \
  --time-map time_map.csv \
  -o timeseries/

binding show-timeseries timeseries/timeseries_position000_channel001.csv \
  --use-time-real -o timeseries/timeseries_position000_channel001.png
```

### 3. Inspect a full-field stack

```bash
binding show DATA -p 0 -c 1 -t 0
```

## Command reference

| Command | Role |
|---|---|
| `spotiflow` | 2D spot detection (ROI batch or single plane) |
| `filter-spots` | Filter Spotiflow CSVs by intensity / size / prob |
| `spot-counts` | Per-cell spot counts over time |
| `plot-lnp` | LNP three-panel figure |
| `timeseries` | ROI mean intensity over time |
| `show-timeseries` | Plot intensity time course |
| `show` | View a raw stack in napari |

## Scripts

| Path | Role |
|------|------|
| `scripts/run_fig5_4s_pipeline.sh` | Full fig5 4 s reanalysis pipeline |
| `scripts/track_spots.py` | Spot tracking / merges |
| `scripts/plot_lnp_early.py` | Early-phase LNP figure helpers |
| `scripts/rd_binding_phases.py` | First-principles RD binding simulation (fig.5 phases) |
| `scripts/fit_rd_binding.py` | Extended RD fit (best-effort) |
| `scripts/rd_results/` | Simulation / fit outputs |
| `scripts/export_ppt_binding_clean.py` | PPT-clean Onpattro vs aiLNP movies + plots → paper presentation |
| `scripts/export_binding_assets.py` | Deck binding stills/movies + `kinetics-real.ts` |

Presentation exporters write into the sibling paper tree by default
(`../lisca-paper/presentation`), override with:

```bash
export LISCA_PAPER_PRESENTATION=/path/to/lisca-paper/presentation
.venv/bin/python scripts/export_ppt_binding_clean.py --plots-only
```

### RD binding theory

```bash
.venv/bin/python scripts/rd_binding_phases.py
.venv/bin/python scripts/fit_rd_binding.py
```

See `scripts/rd_results/README.md`.

## Spotiflow models

| Mode | Default model | Notes |
|---|---|---|
| 2D ROI batch (`--roi-stacks`) | `general` | Point detections; optional `--estimate-params` |
| Single 2D plane | `general` | Full-field single plane |

Multi-z volumetric Spotiflow (`smfish_3d`) lives on branch `3d` only.
