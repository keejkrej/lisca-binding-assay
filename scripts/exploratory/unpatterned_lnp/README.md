# Unpatterned LNP exploratory analysis

Scripts from the A549 60× ND2 analysis, preserving the four detector iterations. `spotiflow/` is the latest run; `picasso_refined/` is the more sensitive Picasso comparison. `dog/` and `picasso/` preserve initial exploratory runs.

Install dependencies from the matching requirements.txt in a Python 3.12 environment. These snapshots include GPU dependencies and are not a replacement for the package dependency specification. Run scripts from any directory. Each defaults to its original results directory under ~/data; override that directory with LNP_RESULTS_DIR. Input ND2 files are read from the results directory's parent. Analyses skip recordings that already contain summary.json; remove that marker to rerun a recording.

For example:

```sh
LNP_RESULTS_DIR=~/data/lnp-tracking-spotiflow python scripts/exploratory/unpatterned_lnp/spotiflow/analyze.py
python scripts/exploratory/unpatterned_lnp/spotiflow/traces.py
python scripts/exploratory/unpatterned_lnp/spotiflow/intensity_distribution.py
```

Spotiflow uses the package's existing pretrained general-model helper; Picasso uses the package's Gaussian-fitting helper. The tracking scripts correct global translation from brightfield and link detections with Trackpy. The Spotiflow and refined Picasso movies show every detection, independent of track-length filtering.

The probe scripts are dataset-specific visual sensitivity experiments. `picasso_refined/check_misses.py PATH_TO_DETECTIONS_CSV` checks four manually reviewed approximate centers in 006 frame zero. It is not comprehensive accuracy validation. Other probe and report scripts expect the adjacent result directories from this analysis. Results README files document detector settings and scientific limitations. No new package API or production pipeline is introduced.

The [btrack comparison](btrack/README.md) relinks the saved Spotiflow detections with a particle-motion model and global gap closing, preserves raw outputs, and screens suspicious distance jumps before plotting.
