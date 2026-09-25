"""
run_cliff_sweep.py
------------------
Focused delay-cliff sweep for the assignment.

Runs:
  2 modes * 3 swarm sizes * 4 cliff delays * 12 seed pairs = 288 jobs

This is meant to fill the gap between the main delay-0 result and the
delay-5 collapse/failure region.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sweep import OUT, MODES, SWARM_SIZES, seed_pairs, run_job  # noqa: E402


CLIFF_DELAYS = (1, 2, 3, 4)
MODEL_TAG = "v2_obstacle_aware_cliff"


def build_jobs(seed_count: int) -> List[Dict]:
    jobs: List[Dict] = []
    for mode in MODES:
        for n in SWARM_SIZES:
            for delay in CLIFF_DELAYS:
                for terrain_seed, deployment_seed in seed_pairs(seed_count):
                    run_id = (
                        f"mode-{mode}_n-{n}_delay-{delay}_"
                        f"terrain-{terrain_seed}_deploy-{deployment_seed}"
                    )
                    jobs.append({
                        "run_id": run_id,
                        "mode": mode,
                        "num_agents": n,
                        "base_delay": delay,
                        "terrain_seed": terrain_seed,
                        "deployment_seed": deployment_seed,
                    })
    return jobs


def write_summary(rows: List[Dict], label: str):
    if not rows:
        return
    path = OUT / "summaries" / f"{MODEL_TAG}_{label}_summary.csv"
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[cliff] summary -> {path}")


def ensure_dirs():
    for name in ("runs", "summaries", "figures", "logs", "configs", "representative"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=288)
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    jobs = build_jobs(args.seed_count)
    start = args.batch_index * args.batch_size
    end = min(start + args.batch_size, len(jobs))
    batch = jobs[start:end]

    print(f"[cliff] total_jobs={len(jobs)} batch={args.batch_index} "
          f"range={start}:{end} batch_size={len(batch)} workers={args.workers}")

    if args.dry_run:
        for job in batch[:12]:
            print(job)
        return

    rows: List[Dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in batch]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[cliff] {i}/{len(batch)} {row['run_id']} {row['status']}")

    elapsed = time.perf_counter() - t0
    label = f"batch_{args.batch_index:03d}"
    write_summary(rows, label)
    log_path = OUT / "logs" / f"{MODEL_TAG}_{label}.log"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "batch_index": args.batch_index,
            "batch_size": len(batch),
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "completed": sum(1 for r in rows if r.get("status") == "completed"),
            "skipped": sum(1 for r in rows if r.get("status") == "skipped_existing"),
        }, f, indent=2)
    print(f"[cliff] batch done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
