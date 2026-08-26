# lisca-binding-assay

## Fleet

PhD work is a multi-repo, multi-machine fleet. Before choosing a machine, cloning, or
moving files, read `~/workspace/phd-notes/standard/README.md`. Status:
`~/workspace/phd-notes/projects/lisca-binding-assay.md`. Prefer `nv5090` for Spotiflow
and figure pipelines.

## Purpose

2D LNP membrane-binding on LISCA ROI crops: Spotiflow → filter → per-cell cluster
**number \(N\)** and **intensity \(I\)**. Review Fig. 5 lives here. Binder-conjugated
LNPs (aiLNP vs Onpattro) plots/movies belong in `~/workspace/ailnp-paper/ppt/`, not in
`lisca-paper`.

3D volumetric workflows are on branch `3d` only, not `main`.

## Commands

```sh
uv sync
uv run binding --help
```

Primary: `spotiflow` → `filter-spots` → `spot-counts` → `plot-lnp`.

Paper/PPT scripts stay in `scripts/` (including `export_ppt_binding_clean.py`,
`rd_binding_phases.py`). Do not copy RD helpers into `lisca-paper`.

## Out of scope

Studio UI, transfection/killing analysis, review-paper prose, 3D on `main`.
