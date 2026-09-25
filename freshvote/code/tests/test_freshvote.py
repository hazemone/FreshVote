import numpy as np
import pytest

from agent import Agent, spawn_agents
from comms import Message
from config import (AgentConfig, ConsensusConfig, FreshnessConfig, MetricsConfig,
                    OptionConfig, TerrainConfig)
from consensus import ConsensusUpdater
from metrics import MetricsRecorder
from terrain import Terrain


def _agent(opinion=0, agent_id=0):
    return Agent(
        id=agent_id,
        position=np.zeros(3),
        velocity=np.zeros(2),
        heading=0.0,
        opinion=opinion,
        quality=1.0,
        base_quality=1.0,
    )


def test_message_age_is_receive_minus_send_time():
    message = Message(0, 1, 1, 0.75, send_time=4, deliver_time=9)
    assert message.age_at(9) == 5
    assert message.age_at(9, clock_offset=-10) == 0


def test_hard_ttl_holds_when_all_messages_are_stale():
    agent = _agent(opinion=0)
    updater = ConsensusUpdater(
        ConsensusConfig(), np.random.default_rng(1),
        FreshnessConfig(policy="hard_ttl", fixed_ttl=2),
    )
    delivered = {0: [Message(1, 0, 1, 1.0, send_time=0, deliver_time=5)]}
    stats = updater.update([agent], delivered, t=5)
    assert agent.opinion == 0
    assert stats.rejected_age_messages == 1
    assert stats.completed_updates == 0


def test_fixed_decay_downweights_older_message():
    updater = ConsensusUpdater(
        ConsensusConfig(), np.random.default_rng(2),
        FreshnessConfig(policy="fixed_decay", fixed_decay_lambda=1.0),
    )
    messages = [
        Message(1, 0, 0, 1.0, send_time=5, deliver_time=5),
        Message(2, 0, 1, 1.0, send_time=0, deliver_time=5),
    ]
    _, weights, _ = updater._policy_weights(0, messages, t=5)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(np.exp(-5.0))


def test_adaptive_policy_does_not_require_future_ages():
    updater = ConsensusUpdater(
        ConsensusConfig(), np.random.default_rng(3),
        FreshnessConfig(policy="adaptive_freshvote", cold_start_observations=1),
    )
    message = Message(1, 0, 1, 0.8, send_time=4, deliver_time=5)
    _, weights, ages = updater._policy_weights(0, [message], t=5)
    assert ages == [1]
    assert weights[0] > 0.0


def test_option_aware_spawn_has_quality_observations_for_both_options():
    terrain_cfg = TerrainConfig(width=30, depth=30, min_density=0.0, max_density=0.0)
    option_cfg = OptionConfig(enabled=True, quality_zero=0.60, quality_one=0.75,
                              measurement_noise_halfwidth=0.0)
    agents = spawn_agents(AgentConfig(num_agents=5), terrain_cfg, Terrain(terrain_cfg),
                          np.random.default_rng(4), option_cfg=option_cfg)
    for agent in agents:
        assert agent.quality_obs_zero == pytest.approx(0.60)
        assert agent.quality_obs_one == pytest.approx(0.75)
        assert agent.quality_for_current_opinion(option_cfg) in (0.60, 0.75)


def test_metrics_classify_stable_correct_consensus():
    recorder = MetricsRecorder(MetricsConfig(consensus_threshold=0.99, persistence_ticks=3), best_option=1)
    agents = [_agent(opinion=1, agent_id=i) for i in range(4)]
    neighbours = {i: [j for j in range(4) if j != i] for i in range(4)}
    for t in range(3):
        recorder.record(agents, neighbours, t, {})
    recorder.finalize()
    summary = recorder.summary()
    assert summary["stable_correct_consensus"]
    assert summary["stable_decision_time"] == 0
    assert not summary["transient_consensus"]
