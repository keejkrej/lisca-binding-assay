# btrack particle relinking of unpatterned A549 LNP recordings

Uses btrack 0.7.0 with a particle random-walk model, local approximate Bayesian updates and global integer-programming gap closing. Inputs are the same saved Spotiflow detections and brightfield drift correction as the Trackpy analysis. This is an exploratory comparison, not a benchmark against manually validated trajectories or a claim of state-of-the-art accuracy.

## Run

Default results directory: ~/data/lnp-tracking-btrack. Override with LNP_RESULTS_DIR; Spotiflow inputs are expected in sibling lnp-tracking-spotiflow. Python 3.12 dependencies are pinned in requirements.txt.

```sh
python scripts/exploratory/unpatterned_lnp/btrack/run_btrack.py --recording 005
python scripts/exploratory/unpatterned_lnp/btrack/run_btrack.py --recording 006
python scripts/exploratory/unpatterned_lnp/btrack/run_btrack.py --recording 007
python scripts/exploratory/unpatterned_lnp/btrack/audit_links.py
python scripts/exploratory/unpatterned_lnp/btrack/plot_results.py
```

The Linux ARM wheel lacked libtracker.so. The local binary was built from upstream v0.7.0 (a3bd947915efe6837936f9db6db88417f0b51b45) with Eigen 3.3.9 (0fd6b4f71dd85b2009ee4d1aeb296e2c11fc9d68), removing x86-only -m64 from compiler flags. The binary is installed only in the analysis virtual environment. It is not committed here.

## Parameters and outputs

A=H=identity (3 states), Q=diag(4,4,0.01), R=diag(0.36,0.36,0.01), initial P=diag(4,4,0.01). Frame indices are model time; actual ND2 elapsed timestamps determine reported track spans. This is isotropic random walk in x/y, not directed constant-velocity extrapolation. Local search radius 6 pixels, max_lost 2, approximate updates; global linking hypotheses only (no branching/merging), spatial threshold 6 pixels and temporal threshold 3 frames. Full per-recording configuration is saved as config.json. Runs logged an optimal global solution under the specified objective, which does not establish correct particle identities.

The first 60 frames of 006 were used to check operation and compare with Trackpy before running all three full recordings. Pilot files and alternate process-noise exploration remain separate from the final directories.

Local and global CSVs preserve native btrack outputs. Predicted/dummy gap positions are exported separately and excluded from observed track lengths, statistics and retained-observation counts. Some input detections are unassigned by btrack and are exported to unassigned_detections.csv. No additional spot detector or score threshold is introduced.

A post-link screen splits raw btrack trajectories wherever displacement between measured positions exceeds 6*sqrt(delta_frames) pixels or observations are more than 4 frames apart. Retained tracks require >=5 real observations AFTER splitting. This conservative heuristic rejects suspicious links but can also split true fast-moving particles. It is not a physics-derived bound. audit.json records the rejected connection count. audited_tracks.csv and audited_track_summary.csv are the screened results; raw global_tracks.csv is retained for inspection. Splits receive new particle IDs and preserve source_btrack_particle.

Plots compare Trackpy with screened btrack results on the same recordings; observed span is not uninterrupted visibility or binding residence time. Comparison is continuity, not identity accuracy. Track fragments do not count unique LNPs. Dense regions, missed detections, focus changes and 8-second sampling can still produce false links and fragmentation. No calibrated diffusion estimate is inferred.

Labeled zoom MP4s display a fixed 300×300-pixel crop at x=150..450, y=350..650, enlarged 2×. To keep labels legible, they show the 25 IDs with most observations in the crop, not every detected punctum. Labels refer to screened track IDs. Colored trails show ten preceding frames; dashed segments indicate a connection across missing observations. Only actual measured centers are circled. Every second acquired frame is shown at 5 fps, so playback is accelerated. Raw data and previous runs are unchanged.

References: https://btrack.readthedocs.io/en/stable/user_guide/configuration.html and https://btrack.readthedocs.io/en/stable/user_guide/large_datasets.html.
