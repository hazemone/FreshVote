"""
dynamics.py
-----------
Agent movement on the 2D obstacle terrain.

Each step every mobile agent:
  1. Computes a candidate new position from its current heading and speed.
  2. Checks if that position is an obstacle.
     - If YES → terrain-induced immobilisation (gap 3): mobile=False, stuck_since=t.
     - If NO  → move, then update heading for the next tick.
  3. In "regroup" scenario, adds a repulsion heading correction that pushes
     agents away from close neighbours — spreads survivors across the terrain
     to maintain coverage after losses (gap 5).
  4. In "static" scenario, no repulsion — each agent keeps its own trajectory.

YOUR 5-FACTOR WORK:
  - apply_speed_variance() → heterogeneous speed across agents
"""

from typing import List
import numpy as np
from agent import Agent
from config import AgentConfig, ScenarioConfig


class DynamicsModel:
    def __init__(self, agent_cfg: AgentConfig, scenario_cfg: ScenarioConfig,
                 rng: np.random.Generator):
        self.agent_cfg = agent_cfg
        self.scenario_cfg = scenario_cfg
        self._rng = rng

    def step(self, agents: List[Agent], terrain, dt: float, t: int):
        """
        Move each mobile agent one timestep and handle obstacle collisions.
        `t` is the current timestep — needed to set stuck_since on immobilisation.
        """
        mobile = [a for a in agents if a.alive and a.mobile]

        for agent in mobile:
            speed = self.apply_speed_variance(self.agent_cfg.base_speed, agent)

            # Candidate displacement along current heading
            dx = speed * np.cos(agent.heading) * dt
            dy = speed * np.sin(agent.heading) * dt
            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy

            # Clamp to terrain bounds before obstacle check
            bx0, bx1, by0, by1 = terrain.world_bounds()
            new_x = float(np.clip(new_x, bx0 + 1, bx1 - 1))
            new_y = float(np.clip(new_y, by0 + 1, by1 - 1))

            # --- GAP 3: obstacle collision → terrain-induced immobilisation ---
            # If the next cell is an obstacle, the agent is physically trapped.
            # It enters ghost state: mobile=False, alive=True, opinion frozen.
            if terrain.is_obstacle(new_x, new_y):
                agent.mobile = False
                agent.stuck_since = t
                agent.velocity = np.zeros(2, dtype=np.float64)
                continue   # don't move this tick

            # Commit the move
            agent.position[0] = new_x
            agent.position[1] = new_y
            agent.position[2] = 0.0           # always flat
            agent.velocity = np.array([dx / dt, dy / dt])

            # --- Heading update ---
            # Base: drift back toward mission heading with heading_noise.
            # This is a weak attraction to the mission direction so agents
            # don't spiral endlessly — they keep advancing forward.
            noise = self.scenario_cfg.heading_noise
            mission = self.scenario_cfg.mission_heading
            # Blend: small pull toward mission + random jitter
            heading_error = _angle_diff(mission, agent.heading)
            agent.heading += 0.1 * heading_error + self._rng.uniform(-noise, noise)

            # --- GAP 5: regroup — repulsion heading correction ---
            # Survivors spread out to maintain coverage. Each mobile agent
            # is nudged away from any alive neighbour within repulsion_radius.
            if self.scenario_cfg.mode == "regroup":
                agent.heading += self._repulsion_correction(agent, mobile)
                agent.heading += self._obstacle_avoidance_correction(agent, terrain)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _repulsion_correction(self, agent: Agent,
                              mobile_agents: List[Agent]) -> float:
        """
        Compute a heading correction that steers agent away from neighbours
        that are too close. Returns a heading delta (radians).

        The correction is the angular direction pointing away from the
        centre-of-mass of nearby neighbours, scaled by how close they are.
        """
        r = self.scenario_cfg.repulsion_radius
        strength = self.scenario_cfg.repulsion_strength
        rx, ry = 0.0, 0.0   # repulsion vector accumulator

        for other in mobile_agents:
            if other.id == agent.id:
                continue
            dx = agent.position[0] - other.position[0]
            dy = agent.position[1] - other.position[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if 0 < dist < r:
                # Weight by inverse distance: closer neighbours push harder
                w = (r - dist) / r
                rx += (dx / dist) * w
                ry += (dy / dist) * w

        if rx == 0.0 and ry == 0.0:
            return 0.0

        # Angle of repulsion vector relative to current heading
        repulsion_angle = np.arctan2(ry, rx)
        delta = _angle_diff(repulsion_angle, agent.heading)
        return strength * delta

    # ------------------------------------------------------------------
    # YOUR FACTOR STUB
    # ------------------------------------------------------------------

    def apply_speed_variance(self, base_speed: float, agent: Agent) -> float:
        """
        TODO: Implement agent speed variance.
        Return the (possibly modified) speed for this agent.
        Default: all agents use base_speed.
        """
        return base_speed

    def _obstacle_avoidance_correction(self, agent: Agent, terrain) -> float:
        """
        Lightweight local obstacle repulsion for regroup mode.
        Prevents coverage-spreading from blindly steering agents into nearby
        obstacle/trap cells.
        """
        radius = self.scenario_cfg.obstacle_avoidance_radius
        strength = self.scenario_cfg.obstacle_avoidance_strength
        if radius <= 0 or strength <= 0:
            return 0.0

        c = terrain.cfg.cell_size
        row, col = terrain._grid_idx(agent.position[0], agent.position[1])
        cell_radius = max(1, int(np.ceil(radius / c)))
        ax, ay = 0.0, 0.0

        for rr in range(row - cell_radius, row + cell_radius + 1):
            if rr < 0 or rr >= terrain.cfg.depth:
                continue
            for cc in range(col - cell_radius, col + cell_radius + 1):
                if cc < 0 or cc >= terrain.cfg.width:
                    continue
                if not terrain.grid[rr, cc]:
                    continue
                ox = (cc + 0.5) * c
                oy = (rr + 0.5) * c
                dx = agent.position[0] - ox
                dy = agent.position[1] - oy
                dist = float(np.sqrt(dx * dx + dy * dy))
                if 0.0 < dist <= radius:
                    w = (radius - dist) / radius
                    ax += (dx / dist) * w
                    ay += (dy / dist) * w

        if ax == 0.0 and ay == 0.0:
            return 0.0
        avoid_angle = np.arctan2(ay, ax)
        return strength * _angle_diff(avoid_angle, agent.heading)


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _angle_diff(target: float, current: float) -> float:
    """Signed shortest angular difference from current to target, in [-π, π]."""
    diff = (target - current + np.pi) % (2 * np.pi) - np.pi
    return diff
