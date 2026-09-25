"""Primary FreshVote evaluation using frozen calibration-selected settings.

The default invocation executes the protocol's 3,840-run test matrix. It uses
seed pairs 21–60, which are disjoint from calibration seeds 1–20.
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from config import (AgentConfig, CommsConfig, Config, FreshnessConfig, MetricsConfig,
                    OptionConfig, SimConfig, TerrainConfig)
from sim import run_simulation


FROZEN_POLICIES = (
    ("baseline", {}),
    ("hard_ttl", {"fixed_ttl": 9}),
    ("fixed_decay", {"fixed_decay_lambda": 0.35}),
    ("adaptive_freshvote", {"adaptive_kappa": 1.5}),
)
DELAYS = (0, 2, 4, 6)
JITTERS = (0, 2, 4)
LOSSES = (0.0, 0.10)


def make_config(job: Dict) -> Config:
    freshness = FreshnessConfig(policy=job["policy"])
    for name, value in job["freshness_parameters"].items():
        setattr(freshness, name, value)
    seed = int(job["seed"])
    return Config(
        terrain=TerrainConfig(seed=30_000 + seed),
        agent=AgentConfig(num_agents=job["agents"]),
        option=OptionConfig(enabled=True, quality_zero=0.60, quality_one=0.75,
                            measurement_noise_halfwidth=0.05),
        comms=CommsConfig(base_delay=job["delay"], delay_jitter_max=job["jitter"],
                          drop_probability=job["loss"]),
        freshness=freshness,
        # The diagnostic sampling frequency does not influence agent trajectories.
        metrics=MetricsConfig(consensus_threshold=0.99, persistence_ticks=300,
                              fiedler_interval=100),
        sim=SimConfig(total_steps=job["steps"], random_seed=seed,
                      deployment_seed=40_000 + seed),
    )


def run_job(job: Dict) -> Dict:
    summary = run_simulation(make_config(job), verbose=False).summary()
    return {
        **{key: value for key, value in job.items() if key != "freshness_parameters"},
        **{f"freshness_{key}": value for key, value in job["freshness_parameters"].items()},
        **summary,
    }


def build_jobs(seed_start: int, seed_count: int, agents: int, steps: int) -> List[Dict]:
    jobs = []
    for policy, parameters in FROZEN_POLICIES:
        for delay in DELAYS:
            for jitter in JITTERS:
                for loss in LOSSES:
                    for seed in range(seed_start, seed_start + seed_count):
                        jobs.append({
                            "policy": policy,
                            "freshness_parameters": parameters,
                            "delay": delay,
                            "jitter": jitter,
                            "loss": loss,
                            "seed": seed,
                            "agents": agents,
                            "steps": steps,
                        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=21)
    parser.add_argument("--seed-count", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--agents", type=int, default=100, help="Pilot override; default is protocol setting.")
    parser.add_argument("--steps", type=int, default=3000, help="Pilot override; default is protocol setting.")
    parser.add_argument("--max-jobs", type=int, default=0, help="Diagnostic limit only; zero runs the full primary matrix.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "results" / "evaluation_full")
    args = parser.parse_args()
    if args.seed_start <= 20:
        raise ValueError("Evaluation seeds must be disjoint from calibration seeds 1–20.")
    if args.seed_count < 1 or args.workers < 1 or args.agents < 2 or args.steps < 1:
        raise ValueError("Invalid execution parameters.")

    jobs = build_jobs(args.seed_start, args.seed_count, args.agents, args.steps)
    expected_full_size = len(FROZEN_POLICIES) * len(DELAYS) * len(JITTERS) * len(LOSSES) * args.seed_count
    if args.max_jobs > 0:
        jobs = jobs[:args.max_jobs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "primary held-out FreshVote evaluation",
        "calibration_seed_range": [1, 20],
        "evaluation_seed_range": [args.seed_start, args.seed_start + args.seed_count - 1],
        "frozen_policies": [{"policy": name, **parameters} for name, parameters in FROZEN_POLICIES],
        "delays": DELAYS,
        "jitters": JITTERS,
        "losses": LOSSES,
        "agents": args.agents,
        "steps": args.steps,
        "expected_full_size": expected_full_size,
        "executed_size": len(jobs),
        "diagnostic_limit": args.max_jobs,
    }
    with (args.output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    rows: List[Dict] = []
    if args.workers == 1:
        for index, job in enumerate(jobs, start=1):
            rows.append(run_job(job))
            print(f"[{index}/{len(jobs)}] {job['policy']} d={job['delay']} j={job['jitter']} p={job['loss']} seed={job['seed']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"[{index}/{len(jobs)}] {row['policy']} d={row['delay']} j={row['jitter']} p={row['loss']} seed={row['seed']}")
    rows.sort(key=lambda row: (row["policy"], row["delay"], row["jitter"], row["loss"], row["seed"]))
    with (args.output_dir / "evaluation_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Completed {len(rows)} of {expected_full_size} planned evaluation runs in {args.output_dir}")


if __name__ == "__main__":
    main()
