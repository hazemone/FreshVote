"""
test_core.py
------------
Unit tests for the binary weighted-voter swarm simulation.

Coverage:
  - Terrain: shape, seed determinism, world bounds, line-of-sight
  - Agent spawning: count, obstacle-avoidance at spawn, opinion/quality ranges
  - Comms primitives: delay arithmetic, noise/loss extremes, distance + ghost decay
  - Consensus: weighted sampling, ghost agents do not update
  - Metrics: Fiedler value on K3 / disconnected, order parameter formula
  - Simulation: smoke run + same-seed determinism
"""

import math

import numpy as np
import pytest

from agent import Agent, spawn_agents
from comms import CommsModel
from config import (
    AgentConfig, CommsConfig, Config, ConsensusConfig, FailureConfig,
    MetricsConfig, ScenarioConfig, SimConfig, TerrainConfig,
)
from consensus import ConsensusUpdater
from metrics import MetricsRecorder
from sim import run_simulation
from terrain import Terrain, generate_obstacle_grid


# ----------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------

def test_terrain_grid_shape():
    grid = generate_obstacle_grid(TerrainConfig(width=50, depth=80, seed=1))
    assert grid.shape == (80, 50)
    assert grid.dtype == bool


def test_terrain_seed_determinism():
    a = generate_obstacle_grid(TerrainConfig(width=50, depth=80, seed=42))
    b = generate_obstacle_grid(TerrainConfig(width=50, depth=80, seed=42))
    np.testing.assert_array_equal(a, b)


def test_terrain_different_seeds_differ():
    a = generate_obstacle_grid(TerrainConfig(width=50, depth=80, seed=1))
    b = generate_obstacle_grid(TerrainConfig(width=50, depth=80, seed=2))
    assert not np.array_equal(a, b)


def test_terrain_world_bounds():
    t = Terrain(TerrainConfig(width=50, depth=80, cell_size=2.0))
    assert t.world_bounds() == (0.0, 100.0, 0.0, 160.0)


def test_terrain_los_clear_in_empty_world():
    cfg = TerrainConfig(width=20, depth=20, min_density=0.0, max_density=0.0, seed=1)
    t = Terrain(cfg)
    assert t.line_of_sight(np.array([1.0, 1.0, 0.0]),
                           np.array([10.0, 10.0, 0.0]))


def test_terrain_los_blocked_by_obstacle():
    cfg = TerrainConfig(width=20, depth=20, min_density=0.0, max_density=0.0, seed=1)
    t = Terrain(cfg)
    t.grid[5, 5] = True
    assert not t.line_of_sight(np.array([1.0, 1.0, 0.0]),
                               np.array([10.0, 10.0, 0.0]))


# ----------------------------------------------------------------------
# Agent spawning
# ----------------------------------------------------------------------

def _spawn(num_agents=20, seed=7):
    terrain_cfg = TerrainConfig(width=50, depth=80, seed=seed)
    agent_cfg = AgentConfig(num_agents=num_agents)
    terrain = Terrain(terrain_cfg)
    rng = np.random.default_rng(0)
    return spawn_agents(agent_cfg, terrain_cfg, terrain, rng), terrain


def test_spawn_count_matches_request():
    agents, _ = _spawn(num_agents=20)
    assert len(agents) == 20


def test_spawned_agents_not_on_obstacles():
    agents, terrain = _spawn(num_agents=15)
    for a in agents:
        assert not terrain.is_obstacle(a.position[0], a.position[1])


def test_spawned_opinions_are_binary():
    agents, _ = _spawn(num_agents=30)
    assert all(a.opinion in (0, 1) for a in agents)


def test_spawned_quality_in_configured_range():
    agents, _ = _spawn(num_agents=30)
    assert all(0.5 <= a.base_quality <= 1.0 for a in agents)


# ----------------------------------------------------------------------
# Comms primitives
# ----------------------------------------------------------------------

def _make_comms(base_delay=0, distance_delay_scale=10.0, distance_decay_scale=25.0,
                noise=0.0, drop=0.0, ghost_window=50):
    cfg = CommsConfig(
        base_delay=base_delay,
        distance_delay_scale=distance_delay_scale,
        distance_decay_scale=distance_decay_scale,
        noise_flip_prob=noise,
        drop_probability=drop,
    )
    fail = FailureConfig(ghost_recognition_window=ghost_window)
    return CommsModel(cfg, fail, np.random.default_rng(0))


def test_apply_delay_zero_distance_returns_base():
    c = _make_comms(base_delay=3, distance_delay_scale=10.0)
    assert c.apply_delay(current_time=10, distance=0.0) == 13


def test_apply_delay_includes_distance_component():
    c = _make_comms(base_delay=2, distance_delay_scale=10.0)
    # ceil(25 / 10) = 3 → 100 + 2 + 3
    assert c.apply_delay(current_time=100, distance=25.0) == 105


def test_noise_zero_never_flips():
    c = _make_comms(noise=0.0)
    assert all(c.apply_noise(1, None, None, 0) == 1 for _ in range(100))


def test_noise_one_always_flips():
    c = _make_comms(noise=1.0)
    assert all(c.apply_noise(1, None, None, 0) == 0 for _ in range(100))


def test_packet_loss_zero_never_drops():
    c = _make_comms(drop=0.0)
    assert not any(c.apply_packet_loss(None, None, 0) for _ in range(100))


def test_packet_loss_one_always_drops():
    c = _make_comms(drop=1.0)
    assert all(c.apply_packet_loss(None, None, 0) for _ in range(100))


def test_distance_decay_at_zero_is_one():
    c = _make_comms(distance_decay_scale=25.0)
    assert c._distance_decay(0.0) == pytest.approx(1.0)


def test_distance_decay_at_scale_is_e_inverse():
    c = _make_comms(distance_decay_scale=25.0)
    assert c._distance_decay(25.0) == pytest.approx(math.exp(-1.0))


def test_distance_decay_far_away_is_zero():
    c = _make_comms(distance_decay_scale=25.0)
    assert c._distance_decay(1e6) == pytest.approx(0.0)


def _agent(stuck_since=-1, mobile=True):
    return Agent(
        id=0, position=np.zeros(3), velocity=np.zeros(2),
        heading=0.0, opinion=0, quality=1.0, base_quality=1.0,
        alive=True, mobile=mobile, stuck_since=stuck_since,
    )


def test_ghost_decay_mobile_agent_is_one():
    c = _make_comms(ghost_window=50)
    assert c._ghost_time_decay(_agent(mobile=True), t=10) == 1.0


def test_ghost_decay_just_stuck_is_one():
    c = _make_comms(ghost_window=50)
    a = _agent(stuck_since=10, mobile=False)
    assert c._ghost_time_decay(a, t=10) == 1.0


def test_ghost_decay_half_window_is_half():
    c = _make_comms(ghost_window=50)
    a = _agent(stuck_since=10, mobile=False)
    assert c._ghost_time_decay(a, t=35) == pytest.approx(0.5)


def test_ghost_decay_past_window_is_zero():
    c = _make_comms(ghost_window=50)
    a = _agent(stuck_since=10, mobile=False)
    assert c._ghost_time_decay(a, t=100) == 0.0


# ----------------------------------------------------------------------
# Consensus
# ----------------------------------------------------------------------

def test_weighted_sampling_picks_only_opinion_with_nonzero_weight():
    """If only one message has positive weight, it must be sampled."""
    upd = ConsensusUpdater(ConsensusConfig(), np.random.default_rng(0))
    msgs = [(0, 0.0), (1, 1.0), (0, 0.0)]
    for _ in range(50):
        assert upd._sample_weighted(msgs) == 1


def test_weighted_sampling_uniform_when_all_zero():
    """All-zero weights should not crash; should return one of the opinions."""
    upd = ConsensusUpdater(ConsensusConfig(), np.random.default_rng(0))
    msgs = [(0, 0.0), (1, 0.0)]
    for _ in range(20):
        assert upd._sample_weighted(msgs) in (0, 1)


def test_ghost_agent_opinion_is_frozen():
    """Ghost (mobile=False) must not have its opinion overwritten."""
    upd = ConsensusUpdater(ConsensusConfig(), np.random.default_rng(0))
    ghost = _agent(stuck_since=5, mobile=False)
    ghost.opinion = 0
    delivered = {0: [(1, 1.0)]}   # would normally flip opinion to 1
    upd.update([ghost], delivered, t=10)
    assert ghost.opinion == 0


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _bare_agent(i):
    return Agent(id=i, position=np.zeros(3), velocity=np.zeros(2),
                 heading=0.0, opinion=0, quality=1.0, base_quality=1.0)


def test_fiedler_complete_triangle_is_three():
    rec = MetricsRecorder(MetricsConfig())
    agents = [_bare_agent(i) for i in range(3)]
    nbrs = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    assert rec._compute_fiedler(agents, nbrs) == pytest.approx(3.0, abs=1e-9)


def test_fiedler_disconnected_is_zero():
    rec = MetricsRecorder(MetricsConfig())
    agents = [_bare_agent(0), _bare_agent(1)]
    nbrs = {0: [], 1: []}
    assert rec._compute_fiedler(agents, nbrs) == pytest.approx(0.0, abs=1e-9)


def test_fiedler_single_node_is_zero():
    rec = MetricsRecorder(MetricsConfig())
    assert rec._compute_fiedler([_bare_agent(0)], {0: []}) == 0.0


def test_order_parameter_formula():
    """|2p - 1|: p=0.5 → 0, p=1 → 1, p=0 → 1."""
    cases = [(0.0, 1.0), (0.25, 0.5), (0.5, 0.0), (0.75, 0.5), (1.0, 1.0)]
    for p, expected in cases:
        assert abs(2 * p - 1) == pytest.approx(expected)


# ----------------------------------------------------------------------
# Simulation: smoke + determinism
# ----------------------------------------------------------------------

def _tiny_config(seed=42, mode="static"):
    cfg = Config(
        terrain=TerrainConfig(width=40, depth=40, seed=seed),
        agent=AgentConfig(num_agents=10),
        scenario=ScenarioConfig(mode=mode),
        comms=CommsConfig(base_delay=0),
        consensus=ConsensusConfig(),
        failure=FailureConfig(),
        metrics=MetricsConfig(record_interval=1, fiedler_interval=10),
        sim=SimConfig(total_steps=50, random_seed=seed, deployment_seed=seed),
    )
    return cfg


def test_simulation_smoke_static():
    metrics = run_simulation(_tiny_config(mode="static"), verbose=False)
    assert len(metrics.time) == 50
    assert metrics.alive_count[0] >= metrics.alive_count[-1]


def test_simulation_smoke_regroup():
    metrics = run_simulation(_tiny_config(mode="regroup"), verbose=False)
    assert len(metrics.time) == 50


def test_simulation_same_seed_same_result():
    """Same config + same seed must produce bit-identical metrics."""
    a = run_simulation(_tiny_config(seed=123), verbose=False)
    b = run_simulation(_tiny_config(seed=123), verbose=False)
    assert a.fraction_one == b.fraction_one
    assert a.order_param == b.order_param
    assert a.alive_count == b.alive_count


def test_simulation_different_seeds_differ():
    a = run_simulation(_tiny_config(seed=1), verbose=False)
    b = run_simulation(_tiny_config(seed=2), verbose=False)
    assert a.fraction_one != b.fraction_one or a.alive_count != b.alive_count
