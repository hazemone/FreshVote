"""
demo.py
-------
Assignment-ready live visualisation.

Defaults:
  - terrain_seed = 42
  - deployment_seed = 42
  - N = 100
  - base communication delay = 200 ticks
  - total_steps = 3000

The demo opens one scenario window at a time. Close the first window and the
next scenario starts automatically.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from config import Config, ScenarioConfig
from terrain import Terrain
from agent import spawn_agents
from comms import CommsModel
from consensus import ConsensusUpdater
from failures import FailureModel
from dynamics import DynamicsModel
from metrics import MetricsRecorder
from viz import LiveVisualizer


ASSIGNMENT_NUM_AGENTS = 100
ASSIGNMENT_BASE_DELAY = 200
ASSIGNMENT_TERRAIN_SEED = 42
ASSIGNMENT_DEPLOYMENT_SEED = 42
ASSIGNMENT_TOTAL_STEPS = 3000


def assignment_config(mode: str) -> Config:
    """Create the assignment demo configuration for a single scenario."""
    cfg = Config()
    cfg.scenario = ScenarioConfig(mode=mode)
    cfg.agent.num_agents = ASSIGNMENT_NUM_AGENTS
    cfg.comms.base_delay = ASSIGNMENT_BASE_DELAY
    cfg.terrain.seed = ASSIGNMENT_TERRAIN_SEED
    cfg.sim.random_seed = ASSIGNMENT_DEPLOYMENT_SEED
    cfg.sim.deployment_seed = ASSIGNMENT_DEPLOYMENT_SEED
    cfg.sim.total_steps = ASSIGNMENT_TOTAL_STEPS
    cfg.metrics.record_interval = 1
    cfg.metrics.fiedler_interval = 25
    return cfg


def run(cfg: Config, label: str):
    deployment_seed = getattr(cfg.sim, "deployment_seed", cfg.sim.random_seed)
    seed_seq = np.random.SeedSequence(deployment_seed)
    spawn_rng, comms_rng, consensus_rng, failure_rng, dynamics_rng = [
        np.random.default_rng(s) for s in seed_seq.spawn(5)
    ]

    terrain = Terrain(cfg.terrain)
    agents = spawn_agents(
        cfg.agent,
        cfg.terrain,
        terrain,
        spawn_rng,
        mission_heading=cfg.scenario.mission_heading,
    )
    comms = CommsModel(cfg.comms, cfg.failure, comms_rng)
    consensus = ConsensusUpdater(cfg.consensus, consensus_rng)
    failures = FailureModel(cfg.failure, failure_rng)
    dynamics = DynamicsModel(cfg.agent, cfg.scenario, dynamics_rng)
    metrics = MetricsRecorder(cfg.metrics)

    vis = LiveVisualizer(terrain, cfg.sim.total_steps, scenario_label=label)

    print(
        f"\n[{label}] Starting | N={cfg.agent.num_agents} "
        f"delay={cfg.comms.base_delay} "
        f"terrain_seed={cfg.terrain.seed} "
        f"deployment_seed={cfg.sim.deployment_seed} "
        f"steps={cfg.sim.total_steps}"
    )
    print(f"[{label}] Close the window to move to the next scenario.")

    for t in range(cfg.sim.total_steps):
        try:
            if not plt.fignum_exists(vis.fig.number):
                print(f"[{label}] Window closed at t={t}.")
                break
        except Exception:
            break

        failures.step(agents, t)

        for agent in agents:
            if (
                agent.alive
                and not agent.mobile
                and agent.stuck_since >= 0
                and (t - agent.stuck_since) >= cfg.failure.ghost_recognition_window
            ):
                agent.alive = False

        neighbours = comms.discover_neighbours(agents, terrain)
        comms.enqueue_messages(agents, neighbours, t)
        delivered = comms.deliver_messages(t)
        consensus.update(agents, delivered, t)
        dynamics.step(agents, terrain, cfg.sim.dt, t)
        metrics.record(agents, neighbours, t)

        vis.update(agents, neighbours, t, metrics)

    s = metrics.summary()
    print(
        f"[{label}] Done p1={s['final_fraction_one']:.3f} "
        f"order={s['final_order_param']:.3f} "
        f"alive={s['final_alive']} mobile={s['final_mobile']} "
        f"ghost={s['final_ghost']}"
    )
    vis.freeze()


if __name__ == "__main__":
    scenarios = [
        ("Static", "static"),
        ("Regroup", "regroup"),
    ]
    for label, mode in scenarios:
        run(assignment_config(mode), label)
