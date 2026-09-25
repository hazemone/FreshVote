"""Main simulation loop for the isolated FreshVote implementation."""

import json
import os
import time
import numpy as np

from agent import spawn_agents
from comms import CommsModel
from config import Config
from consensus import ConsensusUpdater
from dynamics import DynamicsModel
from failures import FailureModel
from metrics import MetricsRecorder
from terrain import Terrain


def run_simulation(cfg: Config, verbose: bool = True) -> MetricsRecorder:
    deployment_seed = getattr(cfg.sim, "deployment_seed", cfg.sim.random_seed)
    seed_seq = np.random.SeedSequence(deployment_seed)
    spawn_rng, comms_rng, consensus_rng, failure_rng, dynamics_rng = [
        np.random.default_rng(stream) for stream in seed_seq.spawn(5)
    ]
    terrain = Terrain(cfg.terrain)
    agents = spawn_agents(cfg.agent, cfg.terrain, terrain, spawn_rng,
                          mission_heading=cfg.scenario.mission_heading,
                          option_cfg=cfg.option)
    comms = CommsModel(cfg.comms, cfg.failure, comms_rng, option_cfg=cfg.option)
    consensus = ConsensusUpdater(cfg.consensus, consensus_rng, cfg.freshness)
    failures = FailureModel(cfg.failure, failure_rng)
    dynamics = DynamicsModel(cfg.agent, cfg.scenario, dynamics_rng)
    metrics = MetricsRecorder(cfg.metrics, best_option=cfg.option.best_option if cfg.option.enabled else None)

    if verbose:
        print(f"[freshvote] policy={cfg.freshness.policy} | {cfg.agent.num_agents} agents | "
              f"{cfg.sim.total_steps} steps | delay={cfg.comms.base_delay} | "
              f"jitter={cfg.comms.delay_jitter_max} | loss={cfg.comms.drop_probability}")
    start = time.time()
    for t in range(cfg.sim.total_steps):
        failures.step(agents, t)
        for agent in agents:
            if (agent.alive and not agent.mobile and agent.stuck_since >= 0
                    and (t - agent.stuck_since) >= cfg.failure.ghost_recognition_window):
                agent.alive = False
        neighbours = comms.discover_neighbours(agents, terrain)
        comms.enqueue_messages(agents, neighbours, t)
        delivered = comms.deliver_messages(t)
        decision_stats = consensus.update(agents, delivered, t).as_dict()
        dynamics.step(agents, terrain, cfg.sim.dt, t)
        metrics.record(agents, neighbours, t, decision_stats)
        if verbose and (t + 1) % 500 == 0:
            print(f"[freshvote] t={t + 1:5d} elapsed={time.time() - start:.1f}s "
                  f"correct={metrics.correct_fraction[-1]:.3f} order={metrics.order_param[-1]:.3f}")
    metrics.finalize()
    if verbose:
        summary = metrics.summary()
        print(f"[freshvote] done={time.time() - start:.1f}s outcome={summary['stable_outcome']} "
              f"transient={summary['transient_consensus']} acceptance={summary['fresh_message_acceptance']:.3f}")
    return metrics


def save_results(metrics: MetricsRecorder, cfg: Config, label: str = "run") -> None:
    out_dir = os.path.join(os.path.dirname(__file__), cfg.sim.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"{label}_metrics.npz"),
        time=np.asarray(metrics.time),
        fraction_one=np.asarray(metrics.fraction_one),
        correct_fraction=np.asarray(metrics.correct_fraction),
        order_param=np.asarray(metrics.order_param),
        variance=np.asarray(metrics.variance),
        fiedler=np.asarray(metrics.fiedler),
        alive_count=np.asarray(metrics.alive_count),
        mobile_count=np.asarray(metrics.mobile_count),
        ghost_count=np.asarray(metrics.ghost_count),
        collapsed=np.asarray(metrics.collapsed),
        fragmented=np.asarray(metrics.fragmented),
        delivered_messages=np.asarray(metrics.delivered_messages),
        accepted_messages=np.asarray(metrics.accepted_messages),
        rejected_age_messages=np.asarray(metrics.rejected_age_messages),
        accepted_age_mean=np.asarray(metrics.accepted_age_mean),
    )
    with open(os.path.join(out_dir, f"{label}_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics.summary(), handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    configuration = Config()
    recorder = run_simulation(configuration, verbose=True)
    save_results(recorder, configuration, label=f"{configuration.freshness.policy}_{configuration.scenario.mode}")
