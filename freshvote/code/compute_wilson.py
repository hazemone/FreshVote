"""
compute_wilson.py
-----------------
Aggregate every v2 sweep CSV (base + cliff) into a single (mode, N, delay) table
and attach Wilson 95% confidence intervals to each consensus rate.

Outputs:
  - new_data/summaries/wilson_consensus_rates.csv    (machine-readable)
  - prints a formatted markdown table to stdout       (paste into the report)
"""

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DIR = ROOT / "new_data" / "summaries"
OUT = SUMMARY_DIR / "wilson_consensus_rates.csv"

# Use only the v2_obstacle_aware (production) sweeps. Skip pre-v2 batches.
INCLUDE_PREFIXES = ("v2_obstacle_aware_batch_", "v2_obstacle_aware_cliff_batch_")


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    """Wilson 95% CI for a binomial proportion. z=1.96 for 95%."""
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


def main():
    rows = load_rows()
    groups = defaultdict(list)
    for r in rows:
        key = (r["mode"], int(r["num_agents"]), int(r["base_delay"]))
        groups[key].append(r)

    out_rows = []
    for key in sorted(groups.keys()):
        mode, n_agents, delay = key
        grp = groups[key]
        n_seeds = len(grp)
        k = sum(1 for r in grp if r["consensus_success"].lower() == "true")
        p, lo, hi = wilson_ci(k, n_seeds)

        orders = [float(r["final_order_param"]) for r in grp]
        alive = [int(r["final_alive"]) for r in grp]
        out_rows.append({
            "mode": mode,
            "num_agents": n_agents,
            "base_delay": delay,
            "n_seeds": n_seeds,
            "consensus_successes": k,
            "consensus_rate": round(p, 4),
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
            "ci_halfwidth": round((hi - lo) / 2, 4),
            "mean_order": round(mean(orders), 4),
            "std_order": round(stdev(orders) if n_seeds > 1 else 0.0, 4),
            "mean_alive": round(mean(alive), 2),
        })

    fields = list(out_rows[0].keys())
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[wilson] wrote {OUT}\n")

    print("| Mode | N | Delay | n | k | rate | 95% Wilson CI | mean order | mean alive |")
    print("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for r in out_rows:
        ci = f"[{r['wilson_lo']*100:.1f}%, {r['wilson_hi']*100:.1f}%]"
        print(f"| {r['mode']} | {r['num_agents']} | {r['base_delay']} | "
              f"{r['n_seeds']} | {r['consensus_successes']} | "
              f"{r['consensus_rate']*100:.1f}% | {ci} | "
              f"{r['mean_order']:.3f} | {r['mean_alive']:.1f} |")


if __name__ == "__main__":
    main()
