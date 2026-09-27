"""
failures.py
-----------
Terrain-induced failure model for the ground swarm.

When an agent is immobilised it enters a GHOST state:
  mobile=False  — can no longer move
  alive=True    — still in the comms graph, still broadcasts stale opinion

The sim loop separately handles the recognition window and sets alive=False
once K steps have elapsed (see sim.py).

YOUR WORK:
  - apply_terrain_immobilisation(): tie immobilisation to actual terrain
    features (steep slope, obstacle proximity, trap zone) once gap-2 terrain
    is in place. For now a Bernoulli stub is used.
  - apply_byzantine_faults(): optional extension — agents that actively
    broadcast wrong opinions (adversarial, not environmental).
"""

from typing import List
import numpy as np
from agent import Agent
from config import FailureConfig


class FailureModel:
    def __init__(self, cfg: FailureConfig, rng: np.random.Generator):
        self.cfg = cfg
        self._rng = rng

    def step(self, agents: List[Agent], t: int):
        """
        Called once per simulation tick before comms/consensus.

        Note: obstacle-collision immobilisation is handled by dynamics.py.
        This method only covers the Bernoulli environmental-event channel
        (mud floods, landslides, etc.) and is disabled by default
        (immobilise_probability=0.0 in config).
        """
        if not self.cfg.enabled or self.cfg.immobilise_probability <= 0.0:
            return

        for agent in agents:
            if not agent.alive or not agent.mobile:
                continue
            self.apply_terrain_immobilisation(agent, t)

    # ------------------------------------------------------------------
    # YOUR FACTOR — implement this
    # ------------------------------------------------------------------

    def apply_terrain_immobilisation(self, agent: Agent, t: int) -> None:
        """
        Decide whether this agent gets stuck this tick.
        Default: Bernoulli draw with probability immobilise_probability.

        When gap-2 terrain is in place, replace the Bernoulli with a
        terrain-feature check, e.g.:
          - slope > threshold  → high immobilisation probability
          - inside obstacle cell → immediate immobilisation

        Side-effects when triggered:
          agent.mobile      = False   (agent stops moving)
          agent.stuck_since = t       (recognition window starts counting)
        """
        if self._rng.random() < self.cfg.immobilise_probability:
            agent.mobile = False
            agent.stuck_since = t

    # ------------------------------------------------------------------
    # Optional extension stub
    # ------------------------------------------------------------------

    def apply_byzantine_faults(self, agent: Agent, t: int) -> None:
        """
        TODO (optional): make a subset of agents actively broadcast
        the wrong opinion. Not needed for the terrain-attrition claim
        but useful as a comparison condition.
        """
        pass
