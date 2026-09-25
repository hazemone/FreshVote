"""
make_extra_figures.py
---------------------
Three additional headline-quality figures squeezed out of the existing
1,008-run dataset — no new simulations needed.

Outputs (all in new_data/figures/):
  - time_series_story.png      :  3-panel order-parameter trajectories for
                                  regroup N=100 at delay = 0, 2, 5.
                                  Shows consensus formation, struggle,
                                  and total failure over simulated time.
  - survival_comparison.png    :  Alive-count-over-time, regroup vs static,
                                  N=100, multiple delays overlaid. Visualises
                                  the survivability separation directly.
  - consensus_heatmap.png      :  Delay (rows) x swarm size (cols) -> consensus
                                  rate, two panels (regroup, static). Compact
                                  summary that complements the line chart.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "new_data" / "runs"
FIG_DIR = ROOT / "new_data" / "figures"
SUMMARY_DIR = ROOT / "new_data" / "summaries"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_SEED = 42
N_SEEDS = 12


def seed_pairs(count=N_SEEDS):
    pairs = [(REFERENCE_SEED, REFERENCE_SEED)]
    for i in range(1, count):
        pairs.append((10_000 + i, 20_000 + i))
    return pairs[:count]


def load_runs(mode: str, n: int, delay: int):
    """Load all seed-pair runs for a given (mode, N, delay) condition."""
    out = []
    for ts, ds in seed_pairs():
        path = RUNS / f"mode-{mode}_n-{n}_delay-{delay}_terrain-{ts}_deploy-{ds}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            out.append({k: data[k] for k in data.files})
    return out


# ------------------------------------------------------------------
# Figure 4 — Time-series story
# ------------------------------------------------------------------

def plot_time_series_story():
    delays_to_show = [0, 2, 5]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)

    for ax, delay in zip(axes, delays_to_show):
        runs = load_runs("regroup", 100, delay)
        for run in runs:
            ax.plot(run["time"], run["order_param"],
                    color="#888888", alpha=0.35, linewidth=1.0)
        if runs:
            stacked = np.vstack([r["order_param"] for r in runs])
            ax.plot(runs[0]["time"], stacked.mean(axis=0),
                    color="#cc3322", linewidth=2.2,
                    label=f"Mean across n={len(runs)} seeds")
        ax.axhline(0.99, color="#3377cc", linestyle=":",
                   linewidth=1.0, alpha=0.7, label="Consensus threshold (0.99)")
        ax.set_ylabel("Order parameter |2p − 1|")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.3)
        ax.set_title(f"Regroup, N = 100, base delay = {delay} ticks",
                     fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Simulation tick (3000 ticks = 300 simulated seconds)")

    fig.suptitle("Order parameter trajectories: consensus formation, struggle, and failure",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = FIG_DIR / "time_series_story.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out}")


# ------------------------------------------------------------------
# Figure 5 — Survival comparison
# ------------------------------------------------------------------

def plot_survival_comparison():
    """
    Plot regroup vs static survival on a single panel, with two delay
    extremes (delay=0 and delay=200) as solid/dashed line styles. The
    overlapping solid/dashed within each mode demonstrates that
    survival is delay-independent.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    style_map = {
        ("regroup", 0):   {"color": "#2a7a3a", "linestyle": "-",  "label": "Regroup, delay = 0"},
        ("regroup", 200): {"color": "#2a7a3a", "linestyle": "--", "label": "Regroup, delay = 200"},
        ("static",  0):   {"color": "#cc3322", "linestyle": "-",  "label": "Static, delay = 0"},
        ("static",  200): {"color": "#cc3322", "linestyle": "--", "label": "Static, delay = 200"},
    }

    for (mode, delay), style in style_map.items():
        runs = load_runs(mode, 100, delay)
        if not runs:
            continue
        stacked = np.vstack([r["alive_count"] for r in runs])
        mean_alive = stacked.mean(axis=0)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)
        t = runs[0]["time"]
        ax.plot(t, mean_alive, color=style["color"],
                linestyle=style["linestyle"], linewidth=2.2,
                label=style["label"])
        if style["linestyle"] == "-":
            ax.fill_between(t, lo, hi, color=style["color"], alpha=0.10)

    ax.set_title("Alive agents over time, N = 100 — survival is delay-independent",
                 fontsize=12)
    ax.set_xlabel("Simulation tick")
    ax.set_ylabel("Alive agent count (mean across n = 12 seeds)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.annotate(
        "Regroup ≈ 100 across all delays",
        xy=(2700, 100), xytext=(2000, 85),
        fontsize=9, color="#2a7a3a",
        arrowprops=dict(arrowstyle="->", color="#2a7a3a", alpha=0.7),
    )
    ax.annotate(
        "Static ≈ 52 across all delays",
        xy=(2900, 52), xytext=(2000, 35),
        fontsize=9, color="#cc3322",
        arrowprops=dict(arrowstyle="->", color="#cc3322", alpha=0.7),
    )

    fig.tight_layout()
    out = FIG_DIR / "survival_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out}")


# ------------------------------------------------------------------
# Figure 6 — Heatmap (delay x N)
# ------------------------------------------------------------------

def plot_heatmap():
    delays = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    sizes = [25, 50, 100]

    grids = {}
    for mode in ["regroup", "static"]:
        grid = np.full((len(delays), len(sizes)), np.nan)
        for i, d in enumerate(delays):
            for j, n in enumerate(sizes):
                runs = load_runs(mode, n, d)
                if runs:
                    k = sum(1 for r in runs if r["order_param"][-1] >= 0.99)
                    grid[i, j] = k / len(runs)
        grids[mode] = grid

    fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), sharey=True)
    cmap = LinearSegmentedColormap.from_list(
        "consensus", ["#0d1424", "#3377cc", "#7ed957", "#ffd166"]
    )

    for ax, mode in zip(axes, ["regroup", "static"]):
        im = ax.imshow(grids[mode], cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"N={n}" for n in sizes])
        ax.set_yticks(range(len(delays)))
        ax.set_yticklabels([str(d) for d in delays])
        ax.set_title(f"{mode.capitalize()}", fontsize=12)
        ax.set_xlabel("Swarm size")
        for i in range(len(delays)):
            for j in range(len(sizes)):
                v = grids[mode][i, j]
                if not np.isnan(v):
                    txt_color = "white" if v < 0.4 else "black"
                    ax.text(j, i, f"{v*100:.0f}%",
                            ha="center", va="center",
                            color=txt_color, fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Base delay (ticks)")

    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.06)
    cbar.set_label("Consensus rate (k / n)", fontsize=10)

    fig.suptitle("Consensus rate heatmap across delay × swarm size", fontsize=13, y=0.99)
    out = FIG_DIR / "consensus_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    plot_time_series_story()
    plot_survival_comparison()
    plot_heatmap()


if __name__ == "__main__":
    main()
