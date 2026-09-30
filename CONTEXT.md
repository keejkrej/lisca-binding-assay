# Binding assay glossary

LiSCA product terms (Workspace, Position, Frame, Timepoint, Interval, Pattern, ROI,
Trace, …): `~/workspace/lisca/CONTEXT.md`. This repo uses them with the same meaning
and adds only the binding-specific terms below.

| Term | Meaning here |
| --- | --- |
| **`time` / `time_real` columns** | Frame index / Timepoint (seconds) in the CSVs this repo writes; `time` follows LiSCA's `time{T}` image-folder naming |
| **\(N\) / \(N_{LNP}\)** | Cumulative Spotiflow cluster count per cell, then mean across cells |
| **\(I\) / \(I_{LNP}\)** | Mean intensity of filtered spots on a cell at \(t\), then mean across cells with ≥1 spot |
| **Onpattro / standard LNP** | Non-binder control formulation |
| **aiLNP** | Binder-conjugated LNP (EGFR). Manuscript claims go to `ailnp-paper` |
