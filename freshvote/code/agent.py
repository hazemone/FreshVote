"""Agent state and initialisation for the FreshVote swarm-decision simulator."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from config import AgentConfig, OptionConfig, TerrainConfig


@dataclass
class Agent:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    opinion: int
    quality: float
    base_quality: float
    alive: bool = True
    mobile: bool = True
    stuck_since: int = -1
    quality_obs_zero: float = 0.0
    quality_obs_one: float = 0.0

    def pos_3d(self) -> np.ndarray:
        return self.position.copy()

    def quality_for_current_opinion(self, option_cfg: OptionConfig) -> float:
        """Return only the sender's local observation for the advertised option."""
        if not option_cfg.enabled:
            return self.base_quality
        return self.quality_obs_one if self.opinion == 1 else self.quality_obs_zero


def spawn_agents(agent_cfg: AgentConfig, terrain_cfg: TerrainConfig, terrain,
                 rng: np.random.Generator, mission_heading: float = 0.0,
                 option_cfg: Optional[OptionConfig] = None) -> List[Agent]:
    """Create agents on free terrain with independent initial opinions."""
    c = terrain_cfg.cell_size
    margin = agent_cfg.spawn_margin
    x_min, x_max = margin, terrain_cfg.width * c - margin
    y_min, y_max = margin, terrain_cfg.depth * c / 3.0
    option_cfg = option_cfg or OptionConfig()

    agents: List[Agent] = []
    for i in range(agent_cfg.num_agents):
        for _ in range(1000):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if not terrain.is_obstacle(x, y):
                break
        else:
            raise RuntimeError("Unable to find a free spawn position after 1000 attempts.")

        heading = mission_heading + rng.uniform(-0.3, 0.3)
        opinion = int(rng.random() < agent_cfg.initial_opinion_one_prob)
        if option_cfg.enabled:
            noise = option_cfg.measurement_noise_halfwidth
            obs_zero = float(np.clip(option_cfg.quality_zero + rng.uniform(-noise, noise), 0.0, 1.0))
            obs_one = float(np.clip(option_cfg.quality_one + rng.uniform(-noise, noise), 0.0, 1.0))
            quality = obs_one if opinion == 1 else obs_zero
        else:
            quality = float(rng.uniform(agent_cfg.quality_min, agent_cfg.quality_max))
            obs_zero = quality
            obs_one = quality

        agents.append(Agent(
            id=i,
            position=np.array([x, y, 0.0], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            heading=heading,
            opinion=opinion,
            quality=quality,
            base_quality=quality,
            quality_obs_zero=obs_zero,
            quality_obs_one=obs_one,
        ))
    return agents
