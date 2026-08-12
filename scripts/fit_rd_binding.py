#!/usr/bin/env python3
"""
Fit the extended LNP binding RD model to experimental median N(t) and I(t).

Extended free parameters (maximize agreement, not minimal physics):
  t_on, tau_mix, S, N_max, block_power, land_gamma, coal_scale, flux_block, D_2d

Method: multi-start local search on a Latin-like grid of seeds, then
Nelder-Mead polish on the best. Mean-field only during optim; final plots
include one stochastic replicate.

Usage (from lisca-binding-assay root):
  .venv/bin/python scripts/fit_rd_binding.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rd_binding_phases import (
    EXP_MERGES,
    K_A,
    OUT_DIR,
    PHASE_BOUNDS_MIN,
    T_END_MIN,
    make_params,
    phase_slopes,
    simulate,
)

FIT_NR, FIT_NZ = 12, 16
FIT_SAMPLE_S = 40.0
FINAL_NR, FINAL_NZ = 20, 28

# Parameter vector:
# [t_on, tau_mix, S, N_max, block_power, land_gamma, log10_D2d, coal_scale, flux_block]
BOUNDS = np.array(
    [
        [0.0, 6.0],  # t_on
        [0.0, 8.0],  # tau_mix
        [0.05, 1.0],  # S
        [100.0, 200.0],  # N_max
        [0.5, 3.5],  # block_power
        [0.25, 2.0],  # land_gamma
        [-16.0, -13.0],  # log10 D_2d
        [0.05, 3.0],  # coal_scale (floor >0 so phase-II merges / I-rise survive)
        [0.05, 1.0],  # flux_block
    ]
)


def load_exp():
    path = Path(__file__).resolve().parent / "_median_trace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    t = np.array([d["t"] for d in data], float)
    N = np.array([d["N"] for d in data], float)
    I = np.array([d["I"] for d in data], float)
    return t, N, I


def vec_to_kwargs(x):
    return dict(
        t_onset_min=float(x[0]),
        tau_mix_min=float(x[1]),
        S=float(x[2]),
        N_max=float(x[3]),
        block_power=float(x[4]),
        land_gamma=float(x[5]),
        D_2d=float(10.0 ** x[6]),
        coal_scale=float(x[7]),
        flux_block=float(x[8]),
    )


def run_from_vec(x, *, nr=FIT_NR, nz=FIT_NZ, progress=False, mode="mean_field", seed=0):
    kw = vec_to_kwargs(x)
    p = make_params(**kw, cluster_mode=mode, seed=seed)
    return simulate(p, nr=nr, nz=nz, sample_every_s=FIT_SAMPLE_S, progress=progress)


def interp_on_exp(run, t_exp):
    N_m = np.interp(t_exp, run["t_min"], run["N"])
    I_raw = np.interp(t_exp, run["t_min"], run["I_mean"])
    I_n = I_raw / I_raw.max() if I_raw.max() > 0 else I_raw
    return N_m, I_n


def objective(x, t_exp, N_exp, I_exp) -> float:
    x = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])
    try:
        run = run_from_vec(x)
    except Exception:
        return 1e6
    N_m, I_m = interp_on_exp(run, t_exp)
    scale_N = max(float(N_exp.max()), 1.0)

    w = np.ones_like(t_exp)
    w[t_exp <= 30] = 1.3
    w[(t_exp > 30) & (t_exp <= 80)] = 1.6
    w[t_exp > 80] = 2.0
    chi_N = float(np.sum(w * ((N_m - N_exp) / scale_N) ** 2))

    # Strong anchors
    for t_a, wa in ((15.0, 2.5), (30.0, 3.0), (50.0, 3.0), (80.0, 4.0), (106.0, 4.0)):
        nm = float(np.interp(t_a, t_exp, N_m))
        ne = float(np.interp(t_a, t_exp, N_exp))
        chi_N += wa * ((nm - ne) / scale_N) ** 2

    m = t_exp >= 5.0
    chi_I = float(np.sum((I_m[m] - I_exp[m]) ** 2))

    # Prefer phase-II dN still positive-ish (soft)
    n30 = float(np.interp(30, t_exp, N_m))
    n80 = float(np.interp(80, t_exp, N_m))
    if n80 + 1e-9 < n30:
        chi_N += 0.5 * ((n30 - n80) / scale_N) ** 2

    # Intensity shape: phase II should rise faster than phase I (exp signature)
    i30 = float(np.interp(30, t_exp, I_m))
    i80 = float(np.interp(80, t_exp, I_m))
    i_end = float(I_m[-1]) if len(I_m) else 0.0
    dI_II = (i80 - i30) / 50.0
    dI_I = (i30 - float(np.interp(10, t_exp, I_m))) / 20.0
    chi_shape = 0.0
    if dI_II + 0.002 < dI_I:  # want II ≥ I roughly
        chi_shape += 2.0 * (dI_I - dI_II) ** 2
    # Late I should be high
    chi_shape += 1.5 * (1.0 - i_end) ** 2

    return chi_N + 0.55 * chi_I + chi_shape


def seed_points(rng: np.random.Generator, n: int = 40) -> list[np.ndarray]:
    """Latin-ish random seeds + a few hand seeds near previous best."""
    seeds = []
    # Hand seeds
    seeds.append(np.array([3.0, 2.0, 0.25, 150.0, 1.5, 0.6, -15.0, 0.3, 0.5]))
    seeds.append(np.array([2.5, 3.0, 0.35, 160.0, 2.0, 0.45, -15.0, 0.2, 0.3]))
    seeds.append(np.array([4.0, 1.5, 0.2, 140.0, 1.2, 0.7, -14.5, 0.5, 0.7]))
    seeds.append(np.array([3.0, 4.0, 0.45, 170.0, 2.5, 0.4, -15.2, 0.1, 0.2]))
    seeds.append(np.array([5.0, 0.5, 0.12, 140.0, 1.0, 0.5, -15.0, 0.5, 1.0]))
    while len(seeds) < n:
        u = rng.random(len(BOUNDS))
        x = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        seeds.append(x)
    return seeds


def fit(t_exp, N_exp, I_exp):
    rng = np.random.default_rng(42)
    seeds = seed_points(rng, n=48)
    print(f"=== Multi-start fit ({len(seeds)} seeds, coarse grid) ===")
    best = {"score": math.inf, "x": seeds[0].copy()}
    t0 = time.time()
    for i, x0 in enumerate(seeds):
        sc = objective(x0, t_exp, N_exp, I_exp)
        if sc < best["score"]:
            best = {"score": sc, "x": x0.copy()}
            kw = vec_to_kwargs(x0)
            print(
                f"  seed {i:02d}  score={sc:.4f}  "
                f"t_on={kw['t_onset_min']:.2f} τ={kw['tau_mix_min']:.2f} "
                f"S={kw['S']:.3f} Nmax={kw['N_max']:.0f} "
                f"bp={kw['block_power']:.2f} γ={kw['land_gamma']:.2f} "
                f"coal={kw['coal_scale']:.2f} fb={kw['flux_block']:.2f}",
                flush=True,
            )
    print(f"  seed scan done in {time.time()-t0:.1f}s  best={best['score']:.4f}")

    # Polish top-K seeds with Nelder-Mead (bound via clip in objective)
    print("=== Nelder-Mead polish on top seeds ===")
    ranked = []
    for x0 in seeds:
        ranked.append((objective(x0, t_exp, N_exp, I_exp), x0))
    ranked.sort(key=lambda z: z[0])
    top = ranked[:8]

    def fun(x):
        return objective(x, t_exp, N_exp, I_exp)

    for j, (sc0, x0) in enumerate(top):
        res = minimize(
            fun,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 120, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True},
        )
        x = np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1])
        sc = fun(x)
        print(f"  polish {j}: {sc0:.4f} → {sc:.4f}  (nit={res.nit})", flush=True)
        if sc < best["score"]:
            best = {"score": sc, "x": x.copy()}

    # One more long polish from best
    print("=== Final polish ===")
    res = minimize(
        fun,
        best["x"],
        method="Nelder-Mead",
        options={"maxiter": 250, "xatol": 5e-4, "fatol": 5e-5, "adaptive": True},
    )
    x = np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1])
    sc = fun(x)
    if sc < best["score"]:
        best = {"score": sc, "x": x.copy()}
    print(f"  final score={best['score']:.4f}")
    return best


def plot_fit(run_prior, run_fit, t_exp, N_exp, I_exp, fit_kw, metrics, out_png, out_svg):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    axN, axI, axS, axR = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    axN.plot(t_exp, N_exp, "k--", lw=1.8, label="exp. median")
    axN.plot(run_prior["t_min"], run_prior["N"], color="#adb5bd", lw=1.3, label="prior")
    axN.plot(run_fit["t_min"], run_fit["N"], color="#c92a2a", lw=2.3, label="fit")
    axN.set_ylabel("N clusters")
    axN.set_title("Cluster count")
    axN.legend(fontsize=8)

    axI.plot(t_exp, I_exp, "k--", lw=1.8, label="exp")
    axI.plot(run_prior["t_min"], run_prior["I_norm"], color="#adb5bd", lw=1.3, label="prior")
    axI.plot(run_fit["t_min"], run_fit["I_norm"], color="#1c7ed6", lw=2.3, label="fit")
    axI.set_ylabel("intensity (norm.)")
    axI.set_title("Mean cluster intensity")
    axI.legend(fontsize=8)

    t_on = fit_kw["t_onset_min"]
    me = t_exp >= t_on
    mf = run_fit["t_min"] >= t_on
    axS.plot(np.sqrt(np.maximum(t_exp[me] - t_on, 0)), N_exp[me], "k--", lw=1.6, label="exp")
    axS.plot(
        np.sqrt(np.maximum(run_fit["t_min"][mf] - t_on, 0)),
        run_fit["N"][mf],
        "o-",
        color="#c92a2a",
        ms=2.5,
        lw=1.5,
        label="fit",
    )
    axS.set_xlabel(r"$\sqrt{t-t_{\mathrm{on}}}$")
    axS.set_ylabel("N")
    axS.set_title("Phase I after onset")
    axS.legend(fontsize=8)

    # Residuals
    N_m, I_m = interp_on_exp(run_fit, t_exp)
    axR.plot(t_exp, N_m - N_exp, color="#c92a2a", lw=1.8, label="ΔN (fit−exp)")
    axR.axhline(0, color="0.5", lw=1)
    axR.set_ylabel("ΔN")
    axR.set_title("N residual")
    axR.legend(fontsize=8)

    for ax in (axN, axI, axR):
        for tb in PHASE_BOUNDS_MIN:
            ax.axvline(tb, color="0.55", ls=":", lw=1)
        ax.set_xlabel("time (min)")
        ax.set_xlim(0, T_END_MIN)

    box = (
        f"Extended fit\n"
        f"t_on={fit_kw['t_onset_min']:.2f}  τ={fit_kw['tau_mix_min']:.2f}\n"
        f"S={fit_kw['S']:.3f}  Nmax={fit_kw['N_max']:.1f}\n"
        f"b_pow={fit_kw['block_power']:.2f}  γ_land={fit_kw['land_gamma']:.2f}\n"
        f"D2d={fit_kw['D_2d']*1e12:.4f} µm²/s\n"
        f"coal={fit_kw['coal_scale']:.2f}  f_block={fit_kw['flux_block']:.2f}\n"
        f"RMSE_N={metrics['rmse_N']:.2f}  RMSE_I={metrics['rmse_I']:.3f}\n"
        f"N30/80/end={metrics['N_30']:.0f}/{metrics['N_80']:.0f}/{metrics['N_end']:.0f}"
    )
    axN.text(
        0.98,
        0.02,
        box,
        transform=axN.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="0.8"),
    )
    fig.suptitle(
        "Extended RD fit vs experimental median (best-effort multi-parameter)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_exp, N_exp, I_exp = load_exp()
    print(f"Loaded exp median: {len(t_exp)} pts  N_end={N_exp[-1]:.1f}")

    best = fit(t_exp, N_exp, I_exp)
    x = best["x"]
    kw = vec_to_kwargs(x)
    print("\n=== Best parameters ===")
    for k, v in kw.items():
        if k == "D_2d":
            print(f"  {k}: {v:.4e} m²/s ({v*1e12:.4f} µm²/s)")
        else:
            print(f"  {k}: {v}")

    print("\n=== Final high-res runs ===")
    prior = run_from_vec(
        np.array([3.0, 0.0, 1.0, 127.0, 1.0, 1.0, -15.0, 1.0, 1.0]),
        nr=FINAL_NR,
        nz=FINAL_NZ,
    )
    run_fit = run_from_vec(x, nr=FINAL_NR, nz=FINAL_NZ, progress=True)
    run_stoch = run_from_vec(
        x, nr=FINAL_NR, nz=FINAL_NZ, progress=True, mode="stochastic", seed=1
    )

    N_m, I_m = interp_on_exp(run_fit, t_exp)
    metrics = {
        "score": best["score"],
        "rmse_N": float(np.sqrt(np.mean((N_m - N_exp) ** 2))),
        "rmse_I": float(np.sqrt(np.mean((I_m - I_exp) ** 2))),
        "N_30": float(np.interp(30, run_fit["t_min"], run_fit["N"])),
        "N_80": float(np.interp(80, run_fit["t_min"], run_fit["N"])),
        "N_end": float(run_fit["N"][-1]),
        "phase_slopes_N": {a: b for a, b in phase_slopes(run_fit["t_min"], run_fit["N"])},
        "phase_slopes_I": {
            a: b for a, b in phase_slopes(run_fit["t_min"], run_fit["I_norm"])
        },
        "merges_stochastic": run_stoch["merges"],
        "exp_merges": EXP_MERGES,
    }
    print(
        f"RMSE_N={metrics['rmse_N']:.2f}  RMSE_I={metrics['rmse_I']:.3f}  "
        f"N(30/80/end)={metrics['N_30']:.1f}/{metrics['N_80']:.1f}/{metrics['N_end']:.1f}  "
        f"exp 70.5/119/126.5"
    )
    print(f"merges: {run_stoch['merges']} vs exp {EXP_MERGES}")

    plot_fit(
        prior,
        run_fit,
        t_exp,
        N_exp,
        I_exp,
        kw,
        metrics,
        OUT_DIR / "rd_fit_overlay.png",
        OUT_DIR / "rd_fit_overlay.svg",
    )

    # Phase I diagnostic
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    m_e = t_exp <= 30
    m_f = run_fit["t_min"] <= 30
    ax.plot(np.sqrt(t_exp[m_e]), N_exp[m_e], "k--", lw=1.6, label="exp")
    ax.plot(
        np.sqrt(run_fit["t_min"][m_f]),
        run_fit["N"][m_f],
        "o-",
        color="#c92a2a",
        ms=3,
        lw=1.5,
        label="fit",
    )
    ax.set_xlabel(r"$\sqrt{t}$ (min$^{1/2}$)")
    ax.set_ylabel("N")
    ax.set_title("Phase I (extended fit)")
    ax.legend()
    fig.savefig(OUT_DIR / "rd_phaseI_sqrt.png", dpi=200)
    plt.close(fig)

    # Merges
    phases = ["I", "II", "III"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    x_pos = np.arange(3)
    exp_v = [EXP_MERGES[p] for p in phases]
    mod_v = [run_stoch["merges"][p] for p in phases]
    ax.bar(x_pos - 0.18, exp_v, 0.35, color="#212529", label="exp 4 s")
    ax.bar(x_pos + 0.18, mod_v, 0.35, color="#c92a2a", label="fit stoch")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["I", "II", "III"])
    ax.set_ylabel("merges")
    ax.legend()
    ax.set_title("Merges by phase")
    fig.savefig(OUT_DIR / "rd_fit_merges.png", dpi=200)
    plt.close(fig)

    payload = {
        "method": "extended multi-start + Nelder-Mead; joint N+I objective",
        "parameters": kw,
        "k_a_m_s": kw["S"] * K_A,
        "metrics": metrics,
        "t_min": run_fit["t_min"].tolist(),
        "N": run_fit["N"].tolist(),
        "I_norm": run_fit["I_norm"].tolist(),
        "M": run_fit["M"].tolist(),
        "bounds": BOUNDS.tolist(),
    }
    (OUT_DIR / "rd_fit_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUT_DIR / 'rd_fit_overlay.png'}")
    print(f"Wrote {OUT_DIR / 'rd_fit_result.json'}")


if __name__ == "__main__":
    main()
