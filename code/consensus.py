"""Weighted-voter decision policies, including the adaptive FreshVote intervention."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple
import numpy as np

from agent import Agent
from comms import DeliveredMap, Message
from config import ConsensusConfig, FreshnessConfig


@dataclass
class DecisionStats:
    delivered_messages: int = 0
    accepted_messages: int = 0
    rejected_age_messages: int = 0
    update_attempts: int = 0
    completed_updates: int = 0
    accepted_age_sum: float = 0.0

    def as_dict(self) -> dict:
        mean_age = self.accepted_age_sum / self.accepted_messages if self.accepted_messages else 0.0
        return {
            "delivered_messages": self.delivered_messages,
            "accepted_messages": self.accepted_messages,
            "rejected_age_messages": self.rejected_age_messages,
            "update_attempts": self.update_attempts,
            "completed_updates": self.completed_updates,
            "accepted_age_sum": self.accepted_age_sum,
            "accepted_age_mean": mean_age,
        }


class ConsensusUpdater:
    """Single-message weighted voter with receiver-side freshness policies."""

    VALID_POLICIES = {"baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"}

    def __init__(self, cfg: ConsensusConfig, rng: np.random.Generator,
                 freshness_cfg: FreshnessConfig | None = None):
        self.cfg = cfg
        self.freshness_cfg = freshness_cfg or FreshnessConfig()
        if self.freshness_cfg.policy not in self.VALID_POLICIES:
            raise ValueError(f"Unknown freshness policy: {self.freshness_cfg.policy}")
        self._rng = rng
        self._age_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(deque)
        self._clock_offsets: Dict[int, int] = {}

    def update(self, agents: List[Agent], delivered: DeliveredMap, t: int) -> DecisionStats:
        stats = DecisionStats()
        for agent in agents:
            if not agent.alive or not agent.mobile:
                continue
            if self._rng.random() >= self.cfg.update_probability:
                continue

            raw_msgs = delivered.get(agent.id, [])
            if not raw_msgs:
                continue
            # Legacy tuple delivery remains supported for baseline unit tests.
            msgs = [m if isinstance(m, Message) else Message(-1, agent.id, m[0], m[1], t, t)
                    for m in raw_msgs]
            stats.update_attempts += 1
            selected, weights, ages = self._policy_weights(agent.id, msgs, t)
            stats.delivered_messages += len(msgs)
            accepted = [idx for idx, weight in enumerate(weights) if weight > 0.0]
            stats.accepted_messages += len(accepted)
            stats.rejected_age_messages += len(msgs) - len(accepted)
            stats.accepted_age_sum += float(sum(ages[idx] for idx in accepted))
            if not selected:
                # Every message was rejected by the local receiver; retain opinion.
                continue
            sampled_opinion = self._sample_weighted(selected, [weight for weight in weights if weight > 0.0])
            agent.opinion = sampled_opinion
            stats.completed_updates += 1
        return stats

    def _policy_weights(self, receiver_id: int, msgs: Sequence[Message], t: int) -> Tuple[List[Message], List[float], List[int]]:
        offset = self._clock_offset(receiver_id)
        ages = [msg.age_at(t, offset) for msg in msgs]
        history = self._age_history[receiver_id]
        for age in ages:
            history.append((t, age))
        min_time = t - self.freshness_cfg.rolling_window_ticks + 1
        while history and history[0][0] < min_time:
            history.popleft()

        policy = self.freshness_cfg.policy
        if policy == "baseline":
            weights = [max(0.0, msg.quality) for msg in msgs]
        elif policy == "hard_ttl":
            weights = [max(0.0, msg.quality) if age <= self.freshness_cfg.fixed_ttl else 0.0
                       for msg, age in zip(msgs, ages)]
        elif policy == "fixed_decay":
            weights = [max(0.0, msg.quality) * float(np.exp(-self.freshness_cfg.fixed_decay_lambda * age))
                       for msg, age in zip(msgs, ages)]
        else:
            weights = self._adaptive_weights(msgs, ages, history)

        selected = [msg for msg, weight in zip(msgs, weights) if weight > 0.0]
        return selected, weights, ages

    def _adaptive_weights(self, msgs: Sequence[Message], ages: Sequence[int], history: Deque[Tuple[int, int]]) -> List[float]:
        if len(history) < self.freshness_cfg.cold_start_observations:
            return [max(0.0, msg.quality) for msg in msgs]
        hist_ages = np.asarray([age for _, age in history], dtype=np.float64)
        median = float(np.median(hist_ages))
        q75, q25 = np.percentile(hist_ages, [75, 25])
        iqr = float(q75 - q25)
        ttl = max(float(self.freshness_cfg.min_ttl), median + self.freshness_cfg.adaptive_kappa * iqr)
        decay = float(np.log(2.0) / max(median, 1.0))
        return [max(0.0, msg.quality) * float(np.exp(-decay * age)) if age <= ttl else 0.0
                for msg, age in zip(msgs, ages)]

    def _clock_offset(self, receiver_id: int) -> int:
        if receiver_id not in self._clock_offsets:
            bound = self.freshness_cfg.clock_skew_max_ticks
            self._clock_offsets[receiver_id] = int(self._rng.integers(-bound, bound + 1)) if bound > 0 else 0
        return self._clock_offsets[receiver_id]

    def _sample_weighted(self, msgs, weights=None) -> int:
        """Sample one positive-weight message; legacy tuple input remains testable."""
        if weights is None:
            opinions = np.array([m[0] for m in msgs], dtype=np.int8)
            weights_array = np.array([m[1] for m in msgs], dtype=np.float64)
        else:
            opinions = np.array([m.opinion for m in msgs], dtype=np.int8)
            weights_array = np.asarray(weights, dtype=np.float64)
        total = weights_array.sum()
        if total <= 0.0:
            idx = self._rng.integers(0, len(opinions))
        else:
            idx = self._rng.choice(len(opinions), p=weights_array / total)
        return int(opinions[idx])
