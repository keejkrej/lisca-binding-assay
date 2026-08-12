# First-principles RD binding simulation (fig.5 phases)

**Repo:** `lisca-binding-assay` (moved from `lisca-paper/scripts/`).  
**Script:** `scripts/rd_binding_phases.py`  
**Rule:** experimental `N(t)`, `I(t)`, and 4 s merge counts are comparison only — never used to fit rates.

```bash
cd ~/workspace/lisca-binding-assay
.venv/bin/python scripts/rd_binding_phases.py
.venv/bin/python scripts/fit_rd_binding.py
```

## Detection vs onset

**Assumption:** each single adsorbed LNP is above the Spotiflow floor used in the
experiment (ref: min intensity 1500, probability 0.25 in the 4 s reanalysis).
Under that assumption, intensity/probability thresholds do **not** filter the
count series — physical \(N\) is compared directly to Spotiflow \(N\).

What *is* applied for a fair timeline comparison is an adsorption **onset**
\(t_\mathrm{on}\approx 3\,\mathrm{min}\): experimental \(t=0\) is acquisition
start, not necessarily first membrane contact (fill/mixing). The membrane sink
is off until \(t_\mathrm{on}\); this is clock alignment, not a rate fit.

```bash
python scripts/rd_binding_phases.py
```

## Model

Axisymmetric finite-volume reaction–diffusion on one array unit cell:

| Piece | Equation / BC |
|-------|----------------|
| Bulk | \(\partial_t c = D\nabla^2 c\) in \((r,z)\), \(0\le r\le R_\mathrm{pitch}\), \(0\le z\le H\) |
| Disk | Langmuir sink on \(r\le R_\mathrm{cell}\) with \(k_a = D/R_\mathrm{LNP}\) |
| Site block | \(\theta = N\,a_\mathrm{block}/A_\mathrm{cell}\) (domain exclusion) shuts off bulk flux as sites fill |
| Glass | no flux for \(r>R_\mathrm{cell}\) |
| Clusters (mean-field) | landing \((1-\phi)\) vs attach; Smoluchowski \(dN/dt = -(K/A)N(N-1)\) |
| Clusters (stochastic) | Poisson landings + Poisson pairwise merges; sizes tracked for \(I=M/N\) |
| Intensity proxy | \(I \propto M/N\), normalised to maximum |

## Upgrades

1. **Site-blocked bulk flux** — domain-scale \(a_\mathrm{block}=\pi(1.5\,\mu\mathrm{m})^2\) so \(N_\mathrm{max}\sim10^2\); adsorption rate → 0 as \(\theta\to1\), without fitting \(k_\mathrm{on}\).
2. **Stochastic coalescence** — integer clusters, Poisson merge events binned into phases I/II/III and compared to experimental 4 s strict merges (0 / 7 / 7).
3. **Adsorption onset** — \(t_\mathrm{on}\) gates the membrane sink for fair clock alignment; no Spotiflow intensity filter under the single-LNP-detectable assumption.

## A priori parameters (not fitted)

| Symbol | Value | Source |
|--------|-------|--------|
| \(D\) | Stokes–Einstein ≈ \(6.1\times10^{-12}\,\mathrm{m^2/s}\) | 100 nm, 37 °C, \(\eta=0.75\,\mathrm{mPa\,s}\) |
| \(c_0\) | \(2\times10^{14}\,\mathrm{m^{-3}}\) | manuscript dose conversion |
| \(H\) | 400 µm | channel height |
| pitch | 120 µm | array unit cell catchment |
| cell | 30 µm square → equal-area disk | micropattern |
| \(D_{2d}\) | 0.001–0.01 µm²/s | membrane NP scale (scenarios) |
| \(k_a\) | \(D/R_\mathrm{LNP}\) | diffusion velocity (near perfect sink) |
| \(k_d\) | 0 | irreversible on 2 h scale |
| \(a_\mathrm{block}\) | \(\pi(1.5\,\mu\mathrm{m})^2\) | domain-exclusion hypothesis |

## Fit (best-effort extended)

```bash
python scripts/fit_rd_binding.py
```

Extended free set (multi-start + Nelder–Mead, joint \(N\)+\(I\)):

| Param | Role |
|-------|------|
| \(t_\mathrm{on}\), \(\tau_\mathrm{mix}\) | hard onset + soft fill ramp |
| \(S\) | sticking vs DL \(k_a\) |
| \(N_\mathrm{max}\) | spot capacity |
| `block_power` | delayed flux blocking \(\theta=(N/N_\max)^p\) |
| `land_gamma` | nucleation \(p_\mathrm{new}=(1-N/N_\max)^\gamma\) |
| \(D_{2d}\), `coal_scale` | coalescence |
| `flux_block` | strength of site suppression on bulk flux |

Latest best-fit: `rd_fit_result.json` / `rd_fit_overlay.png`.

## Outputs

| File | Content |
|------|---------|
| `rd_binding_phases.{png,svg}` | N, I, M, bulk depletion vs exp. median |
| `rd_phaseI_sqrt.png` | \(N\) vs \(\sqrt{t}\) diagnostic (fit when fit script last ran) |
| `rd_merges_by_phase.{png,svg}` | merge counts by phase vs 4 s tracking |
| `rd_binding_baseline.json` | numeric primary series + merges |
| `rd_fit_overlay.{png,svg}` | prior vs restricted fit vs exp |
| `rd_fit_merges.png` | stochastic merges at fit params |
| `rd_fit_result.json` | best-fit parameters + metrics |

## What the comparison shows

1. **Phase I is transport-controlled.** Axisymmetric RD gives \(N(30\,\mathrm{min})\) of the same order as the measured median (~70), between planar Ward–Tordai (~21) and sphere Smoluchowski (~460).
2. **Lateral supply is required.** Only ~72 LNPs sit above one cell column; the pitch column holds ~1152.
3. **Phase II intensity rise is generic.** Arrival + coalescence/attachment raises \(M/N\) while slowing new spots.
4. **Phase III needs site blocking.** Hard packing cannot explain \(N_\mathrm{sat}\sim126\); domain exclusion does, and site-blocked flux also flattens late intensity.
5. **Merges concentrate in II–III.** Stochastic pairwise rates \(\propto N(N-1)\) put essentially zero merges in the dilute phase-I window — same ordering as the 4 s experiment.
