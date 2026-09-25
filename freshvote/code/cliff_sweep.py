"""
cliff_sweep.py
--------------
Pin the consensus cliff between delay=0 (83% consensus) and delay=5 (0%).

Runs regroup N=100 at delay in {1, 2, 3, 4}, 12 seeds, 48 total runs.
Reuses run_job from run_sweep.py (same model, same skip-existing logic,
same output paths under new_data/runs/ and new_data/configs/).
"""

from __future__ import annotations
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sweep import run_job, ensure_dirs, seed_pairs, OUT

CLIFF_DELAYS = (1, 2, 3, 4)
SEED_COUNT = 12
WORKERS = 8
BATCH_TAG = "cliff_regroup_n100"


def build_cliff_jobs():
    jobs = []
    for delay in CLIFF_DELAYS:
        for terrain_seed, deployment_seed in seed_pairs(SEED_COUNT):
            run_id = (
                f"mode-regroup_n-100_delay-{delay}_"
                f"terrain-{terrain_seed}_deploy-{deployment_seed}"
            )
            jobs.append({
                "run_id": run_id,
                "mode": "regroup",
                "num_agents": 100,
                "base_delay": delay,
                "terrain_seed": terrain_seed,
                "deployment_seed": deployment_seed,
            })
    return jobs


def main():
    ensure_dirs()
    jobs = build_cliff_jobs()
    print(f"[cliff] {len(jobs)} jobs across delays={CLIFF_DELAYS}, workers={WORKERS}")

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), start=1):
            row = f.result()
            rows.append(row)
            print(f"[cliff] {i}/{len(jobs)} {row['run_id']} {row['status']}")

    summary_path = OUT / "summaries" / f"{BATCH_TAG}_summary.csv"
    fields = sorted({k for r in rows for k in r.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - t0
    log_path = OUT / "logs" / f"{BATCH_TAG}.log"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "batch": BATCH_TAG,
            "delays": list(CLIFF_DELAYS),
            "seeds": SEED_COUNT,
            "workers": WORKERS,
            "elapsed_seconds": elapsed,
            "completed": sum(1 for r in rows if r.get("status") == "completed"),
            "skipped": sum(1 for r in rows if r.get("status") == "skipped_existing"),
        }, f, indent=2)

    print(f"[cliff] done in {elapsed:.1f}s -> {summary_path}")


if __name__ == "__main__":
    main()
