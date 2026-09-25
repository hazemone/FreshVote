"""
make_headline_figure.py
-----------------------
Headline figure for the assignment report.

Produces:
  new_data/figures/headline_consensus_vs_delay.png
  new_data/figures/headline_consensus_vs_delay_zoom.png  (delay 0-5 only)

For each (mode, N) panel:
  - line: consensus rate vs delay
  - error bars: Wilson 95% CI
  - light dots: seed-level binary outcomes (jittered for readability)
  - n=12 noted in caption
"""

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "new_data" / "summaries"
FIG_DIR = ROOT / "new_data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INCLUDE_PREFIXES = ("v2_obstacle_aware_batch_", "v2_obstacle_aware_cliff_batch_")


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_rows():
    rows = []
    for path in sorted(SUMMARY_DIR.glob("*.csv")):
        if not path.name.startswith(INCLUDE_PREFIXES):
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def aggregate(rows):
    """Returns nested dict: data[mode][n_agents][delay] = list[bool successes]."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        mode = r["mode"]
        n = int(r["num_agents"])
        d = int(r["base_delay"])
        success = r["consensus_success"].lower() == "true"
        data[mode][n][d].append(success)
    return data


def plot_panel(ax, delays, successes_by_delay, mode, n_agents):
    rates, los, his = [], [], []
    rng = np.random.default_rng(0)
    for d in delays:
        successes = successes_by_delay.get(d, [])
        n_seeds = len(successes)
        k = sum(successes)
        p, lo, hi = wilson_ci(k, n_seeds)
        rates.append(p)
        los.append(max(0.0, p - lo))
        his.append(max(0.0, hi - p))

        # seed-level dots, jittered horizontally
        for s in successes:
            jitter = rng.uniform(-0.15, 0.15)
            ax.plot(d + jitter, 1.0 if s else 0.0,
                    "o", color="#888888", alpha=0.35, markersize=4,
                    markeredgewidth=0)

    ax.errorbar(delays, rates, yerr=[los, his],
                fmt="-o", color="#cc3322", capsize=3, linewidth=1.6,
                markersize=5, label="Consensus rate (Wilson 95% CI)")

    ax.set_title(f"{mode.capitalize()}, N={n_agents}", fontsize=11)
    ax.set_xlabel("Base delay (ticks)")
    ax.set_ylabel("Consensus rate")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def make_figure(rows, delays, fname, suptitle):
    data = aggregate(rows)
    modes = ["regroup", "static"]
    sizes = [25, 50, 100]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for i, mode in enumerate(modes):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            plot_panel(ax, delays, data[mode][n], mode, n)

    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.text(0.5, 0.01,
             "Dots: per-seed outcomes (n=12 seeds per condition, jittered). "
             "Bars: Wilson 95% CI on consensus rate.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path = FIG_DIR / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out_path}")


def main():
    rows = load_rows()

    headline_delays = [0, 1, 2, 3, 4, 5, 10]
    make_figure(
        rows, headline_delays,
        "headline_consensus_vs_delay.png",
        "Consensus rate vs communication delay (regroup vs static, N=25/50/100)",
    )

    zoom_delays = [0, 1, 2, 3, 4, 5]
    make_figure(
        rows, zoom_delays,
        "headline_consensus_vs_delay_zoom.png",
        "Consensus rate vs communication delay — zoom on cliff (delay 0–5)",
    )


if __name__ == "__main__":
    main()
