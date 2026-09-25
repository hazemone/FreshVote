"""
run_sweep.py
------------
CPU-parallel sweep runner for the A-class assignment path.

All new outputs are written under ../new_data so old data/ and figures/ remain
untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, ScenarioConfig  # noqa: E402
from sim import run_simulation  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "new_data"
MODEL_TAG = "v2_obstacle_aware"

MODES = ("static", "regroup")
SWARM_SIZES = (25, 50, 100)
DELAYS = (0, 5, 10, 20, 30, 50, 75, 100, 150, 200)
REFERENCE_SEED = 42


def seed_pairs(count: int) -> List[tuple[int, int]]:
    """Return deterministic (terrain_seed, deployment_seed) pairs."""
    pairs = [(REFERENCE_SEED, REFERENCE_SEED)]
    for i in range(1, count):
        pairs.append((10_000 + i, 20_000 + i))
    return pairs[:count]


def build_jobs(seed_count: int) -> List[Dict]:
    jobs: List[Dict] = []
    for mode in MODES:
        for n in SWARM_SIZES:
            for delay in DELAYS:
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


def first_consensus_time(order_param: np.ndarray, threshold: float = 0.99):
    hits = np.flatnonzero(order_param >= threshold)
    if len(hits) == 0:
        return None
    return int(hits[0])


def summary_from_arrays(job: Dict, arrays: Dict, runtime_seconds=None,
                        status: str = "completed") -> Dict:
    order = arrays["order_param"]
    consensus_t = first_consensus_time(order)
    collapsed = arrays["collapsed"]
    collapse_hits = np.flatnonzero(collapsed)
    collapse_t = int(collapse_hits[0]) if len(collapse_hits) else None
    return {
        "run_id": job["run_id"],
        "status": status,
        "mode": job["mode"],
        "num_agents": job["num_agents"],
        "base_delay": job["base_delay"],
        "terrain_seed": job["terrain_seed"],
        "deployment_seed": job["deployment_seed"],
        "steps": int(len(arrays["time"])),
        "consensus_timestep": consensus_t,
        "consensus_success": bool(order[-1] >= 0.99),
        "collapse_detected": bool(len(collapse_hits)),
        "collapse_timestep": collapse_t,
        "final_fraction_one": float(arrays["fraction_one"][-1]),
        "final_order_param": float(arrays["order_param"][-1]),
        "final_variance": float(arrays["variance"][-1]),
        "final_fiedler": float(arrays["fiedler"][-1]),
        "final_fragmented": bool(arrays["fragmented"][-1]),
        "final_alive": int(arrays["alive_count"][-1]),
        "final_mobile": int(arrays["mobile_count"][-1]),
        "final_ghost": int(arrays["ghost_count"][-1]),
        "runtime_seconds": runtime_seconds,
    }


def run_job(job: Dict) -> Dict:
    t0 = time.perf_counter()

    cfg = Config()
    cfg.scenario = ScenarioConfig(mode=job["mode"])
    cfg.agent.num_agents = int(job["num_agents"])
    cfg.comms.base_delay = int(job["base_delay"])
    cfg.terrain.seed = int(job["terrain_seed"])
    cfg.sim.deployment_seed = int(job["deployment_seed"])
    cfg.sim.random_seed = int(job["deployment_seed"])
    cfg.sim.total_steps = 3000
    cfg.metrics.record_interval = 1
    cfg.metrics.fiedler_interval = 25
    cfg.sim.output_dir = "../new_data"

    run_path = OUT / "runs" / f"{job['run_id']}.npz"
    cfg_path = OUT / "configs" / f"{job['run_id']}.json"
    if run_path.exists() and cfg_path.exists():
        with np.load(run_path) as data:
            arrays = {k: data[k] for k in data.files}
        return summary_from_arrays(job, arrays, status="skipped_existing")

    metrics = run_simulation(cfg, verbose=False)
    summary = metrics.summary()

    arrays = {
        "time": np.array(metrics.time, dtype=np.int32),
        "fraction_one": np.array(metrics.fraction_one, dtype=np.float32),
        "order_param": np.array(metrics.order_param, dtype=np.float32),
        "variance": np.array(metrics.variance, dtype=np.float32),
        "fiedler": np.array(metrics.fiedler, dtype=np.float32),
        "alive_count": np.array(metrics.alive_count, dtype=np.int16),
        "mobile_count": np.array(metrics.mobile_count, dtype=np.int16),
        "ghost_count": np.array(metrics.ghost_count, dtype=np.int16),
        "collapsed": np.array(metrics.collapsed, dtype=bool),
        "fragmented": np.array(metrics.fragmented, dtype=bool),
    }
    np.savez(run_path, **arrays)

    config_dump = asdict(cfg)
    config_dump["run_id"] = job["run_id"]
    config_dump["model_version"] = "weighted_ghost_time_distance_obstacle_aware_regroup_cpu_v2"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    elapsed = time.perf_counter() - t0
    return summary_from_arrays(job, arrays, runtime_seconds=round(elapsed, 4))


def ensure_dirs():
    for name in ("runs", "summaries", "figures", "logs", "configs", "representative"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def write_summary(rows: List[Dict], batch_index: int):
    if not rows:
        return
    path = OUT / "summaries" / f"{MODEL_TAG}_batch_{batch_index:03d}_summary.csv"
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--seed-count", type=int, default=12)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    jobs = build_jobs(args.seed_count)
    start = args.batch_index * args.batch_size
    end = min(start + args.batch_size, len(jobs))
    batch = jobs[start:end]

    print(f"[sweep] total_jobs={len(jobs)} batch={args.batch_index} "
          f"range={start}:{end} batch_size={len(batch)} workers={args.workers}")

    if args.dry_run:
        for job in batch[:10]:
            print(job)
        return

    rows: List[Dict] = []
    log_path = OUT / "logs" / f"{MODEL_TAG}_batch_{args.batch_index:03d}.log"
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_job, job) for job in batch]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[sweep] {i}/{len(batch)} {row['run_id']} {row['status']}")

    write_summary(rows, args.batch_index)
    elapsed = time.perf_counter() - t0
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({
            "batch_index": args.batch_index,
            "batch_size": len(batch),
            "workers": args.workers,
            "elapsed_seconds": elapsed,
            "completed": sum(1 for r in rows if r.get("status") == "completed"),
            "skipped": sum(1 for r in rows if r.get("status") == "skipped_existing"),
        }, f, indent=2)
    print(f"[sweep] batch done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
