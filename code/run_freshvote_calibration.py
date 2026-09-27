"""Held-out parameter calibration for FreshVote.

Runs only the fixed representative stress condition from the approved manifest.
The output is a training-only CSV; calibration seeds must never enter evaluation.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

from config import (AgentConfig, CommsConfig, Config, FreshnessConfig, MetricsConfig,
                    OptionConfig, SimConfig, TerrainConfig)
from sim import run_simulation


REPRESENTATIVE_STRESS = {"delay": 4, "jitter": 2, "loss": 0.10}
CANDIDATES = (
    [("baseline", "none", 0.0)]
    + [("hard_ttl", "fixed_ttl", float(value)) for value in (3, 5, 7, 9)]
    + [("fixed_decay", "fixed_decay_lambda", float(value)) for value in (0.10, 0.20, 0.35, 0.50)]
    + [("adaptive_freshvote", "adaptive_kappa", float(value)) for value in (0.5, 1.0, 1.5, 2.0)]
)


def make_config(policy: str, parameter: str, value: float, seed: int,
                agents: int = 100, steps: int = 3000) -> Config:
    freshness = FreshnessConfig(policy=policy)
    if parameter != "none":
        setattr(freshness, parameter, int(value) if parameter == "fixed_ttl" else value)
    return Config(
        terrain=TerrainConfig(seed=30_000 + seed),
        agent=AgentConfig(num_agents=agents),
        option=OptionConfig(enabled=True, quality_zero=0.60, quality_one=0.75,
                            measurement_noise_halfwidth=0.05),
        comms=CommsConfig(base_delay=REPRESENTATIVE_STRESS["delay"],
                          delay_jitter_max=REPRESENTATIVE_STRESS["jitter"],
                          drop_probability=REPRESENTATIVE_STRESS["loss"]),
        freshness=freshness,
        # Connectivity is a non-primary diagnostic in FreshVote; this does not affect dynamics.
        metrics=MetricsConfig(consensus_threshold=0.99, persistence_ticks=300, fiedler_interval=100),
        sim=SimConfig(total_steps=steps, random_seed=seed, deployment_seed=40_000 + seed),
    )


def run_job(job: Dict) -> Dict:
    cfg = make_config(job["policy"], job["parameter"], job["value"], job["seed"],
                      agents=job.get("agents", 100), steps=job.get("steps", 3000))
    summary = run_simulation(cfg, verbose=False).summary()
    return {**job, **REPRESENTATIVE_STRESS, **summary}


def select_candidate(rows: List[Dict]) -> List[Dict]:
    """Apply the manifest rule: max stable-correct rate, then lowest median time."""
    groups = {}
    for row in rows:
        key = (row["policy"], row["parameter"], float(row["value"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (policy, parameter, value), group in sorted(groups.items()):
        stable = [row for row in group if str(row["stable_correct_consensus"]).lower() == "true"]
        times = sorted(float(row["stable_decision_time"]) for row in stable if row["stable_decision_time"] not in ("", None))
        median_time = times[len(times) // 2] if times else float("inf")
        output.append({
            "policy": policy,
            "parameter": parameter,
            "value": value,
            "runs": len(group),
            "stable_correct_rate": len(stable) / len(group),
            "median_stable_decision_time_among_successes": median_time,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--agents", type=int, default=100, help="Pilot override; default is locked evaluation size.")
    parser.add_argument("--steps", type=int, default=3000, help="Pilot override; default is locked evaluation budget.")
    parser.add_argument("--max-jobs", type=int, default=0, help="Diagnostic limit only; zero runs the complete calibration matrix.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "results" / "calibration")
    args = parser.parse_args()
    if args.seeds < 1 or args.workers < 1 or args.agents < 2 or args.steps < 1:
        raise ValueError("--seeds/workers/steps must be positive and --agents must be at least two.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"policy": policy, "parameter": parameter, "value": value, "seed": seed,
         "agents": args.agents, "steps": args.steps}
        for policy, parameter, value in CANDIDATES
        for seed in range(1, args.seeds + 1)
    ]
    if args.max_jobs > 0:
        jobs = jobs[:args.max_jobs]
    rows: List[Dict] = []
    if args.workers == 1:
        for index, job in enumerate(jobs, start=1):
            rows.append(run_job(job))
            print(f"[{index}/{len(jobs)}] {job['policy']} {job['parameter']}={job['value']} seed={job['seed']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"[{index}/{len(jobs)}] {row['policy']} {row['parameter']}={row['value']} seed={row['seed']}")
    rows.sort(key=lambda row: (row["policy"], row["parameter"], row["value"], row["seed"]))
    with (output_dir / "calibration_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    selection = select_candidate(rows)
    with (output_dir / "calibration_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)
    print(f"Completed {len(rows)} calibration runs in {output_dir}")


if __name__ == "__main__":
    main()
