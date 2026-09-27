"""Experiment 2 time-series figure. Re-runs one held-out condition with series recording.

Runs are deterministic, so these trajectories are the same runs as in evaluation_runs.csv
(checked below against the stored post-swap accuracy).
"""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "code"))
import style; style.apply()
from sim_cache import CacheConfig, run_cache

man = json.loads((ROOT / "results/cache_experiment/evaluation_manifest.json").read_text())
BEACON, CH = int(sys.argv[1]) if len(sys.argv) > 1 else 30, sys.argv[2] if len(sys.argv) > 2 else "severe"
SEEDS = range(201, 241)


def job(args):
    pol, seed = args
    kw = dict(man["common"], **man["channels"][CH], beacon_period=BEACON, policy=pol, seed=seed, **man["frozen"])
    r = run_cache(CacheConfig(**kw), keep_series=True)
    return pol, seed, r["post_mean_correct"], r["series_t"], r["series_c"]


def plot(t, series, man):
    swap = man["common"]["swap_tick"] * 0.1
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    for p in style.POLICIES:
        m = pd.Series(series[p].mean(0))
        pre = t < swap
        # 5-sample (5 s) centred moving average, computed separately before and after the swap
        sm = pd.concat([m[pre].rolling(5, center=True, min_periods=1).mean(),
                        m[~pre].rolling(5, center=True, min_periods=1).mean()])
        ax.plot(t[pre], sm[pre], color=style.COLORS[p], ls=style.LINES[p], lw=1.2, label=style.LABELS[p])
        ax.plot(t[~pre], sm[~pre], color=style.COLORS[p], ls=style.LINES[p], lw=1.2)
    ax.axvline(swap, color="#8a8a86", lw=0.8, ls=":")
    ax.text(swap + 4, 0.03, "quality swap", fontsize=7, color="#5f5e5a")
    ax.set_xlim(0, 600); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Fraction on best option")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False, handlelength=1.6, columnspacing=0.8)
    fig.savefig(ROOT / "figures/fig_exp2_series.pdf"); fig.savefig(ROOT / "figures/fig_exp2_series.png", dpi=300)


if __name__ == "__main__" and "--plot-only" in sys.argv:
    z = np.load(ROOT / f"analysis/exp2/series_B{BEACON}_{CH}.npz")
    plot(z["t"], {p: z[p] for p in style.POLICIES}, man); sys.exit()

if __name__ == "__main__":
    jobs = [(p, s) for p in style.POLICIES for s in SEEDS]
    with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as ex:
        res = list(ex.map(job, jobs))
    ev = pd.read_csv(ROOT / "results/cache_experiment/evaluation_runs.csv")
    ev = ev[(ev.beacon_period == BEACON) & (ev.base_delay == man["channels"][CH]["base_delay"])].set_index(["policy", "seed"])
    assert all(abs(ev.loc[(p, s), "post_mean_correct"] - pm) < 1e-12 for p, s, pm, _, _ in res), "rerun mismatch"
    series = {p: np.array([c for q, s, pm, t, c in res if q == p]) for p in style.POLICIES}
    t = np.array(res[0][3]) * 0.1
    np.savez(ROOT / f"analysis/exp2/series_B{BEACON}_{CH}.npz", t=t, **series)
    plot(t, series, man)
    print("ok")
