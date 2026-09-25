"""Non-publication smoke test for FreshVote integration.

This script checks that every policy executes, produces the new metrics, and
writes a traceable CSV. It is not the preregistered evaluation experiment.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from config import (AgentConfig, CommsConfig, Config, FreshnessConfig, MetricsConfig,
                    OptionConfig, SimConfig, TerrainConfig)
from sim import run_simulation


POLICIES = ["baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"]


def make_config(policy: str, delay: int, jitter: int, loss: float, seed: int) -> Config:
    return Config(
        terrain=TerrainConfig(width=60, depth=100, seed=10_000 + seed,
                              min_density=0.0, max_density=0.002),
        agent=AgentConfig(num_agents=20),
        option=OptionConfig(enabled=True, quality_zero=0.60, quality_one=0.75,
                            measurement_noise_halfwidth=0.05),
        comms=CommsConfig(base_delay=delay, delay_jitter_max=jitter,
                          drop_probability=loss),
        freshness=FreshnessConfig(policy=policy, fixed_ttl=5,
                                  fixed_decay_lambda=0.20, adaptive_kappa=1.0),
        metrics=MetricsConfig(consensus_threshold=0.95, persistence_ticks=20,
                              fiedler_interval=10),
        sim=SimConfig(total_steps=300, random_seed=seed, deployment_seed=20_000 + seed),
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "results" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for policy in POLICIES:
        for delay in (0, 4):
            for jitter in (0, 2):
                for loss in (0.0, 0.10):
                    for seed in (1, 2):
                        cfg = make_config(policy, delay, jitter, loss, seed)
                        metrics = run_simulation(cfg, verbose=False)
                        summary = metrics.summary()
                        rows.append({
                            "policy": policy,
                            "delay": delay,
                            "jitter": jitter,
                            "loss": loss,
                            "seed": seed,
                            **summary,
                        })
    with (output_dir / "smoke_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "smoke_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "purpose": "integration smoke test only; not publication evidence",
            "policies": POLICIES,
            "delays": [0, 4],
            "jitters": [0, 2],
            "losses": [0.0, 0.10],
            "seeds": [1, 2],
            "total_runs": len(rows),
        }, handle, indent=2, sort_keys=True)
    print(f"Smoke test completed: {len(rows)} runs written to {output_dir}")


if __name__ == "__main__":
    main()
