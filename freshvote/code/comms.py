"""Communication model with timestamp-preserving delivery for FreshVote."""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple
import math
import numpy as np

from agent import Agent
from config import CommsConfig, FailureConfig, OptionConfig


@dataclass(frozen=True)
class Message:
    sender_id: int
    receiver_id: int
    opinion: int
    quality: float
    send_time: int
    deliver_time: int

    def age_at(self, receive_time: int, clock_offset: int = 0) -> int:
        """Receiver-visible non-negative age in integer simulation ticks."""
        return max(0, int(receive_time + clock_offset - self.send_time))


DeliveredMap = Dict[int, List[Message]]


class CommsModel:
    def __init__(self, cfg: CommsConfig, failure_cfg: FailureConfig,
                 rng: np.random.Generator, option_cfg: OptionConfig | None = None):
        self.cfg = cfg
        self.failure_cfg = failure_cfg
        self.option_cfg = option_cfg or OptionConfig()
        self._queue: List[Message] = []
        self._rng = rng

    def discover_neighbours(self, agents: List[Agent], terrain) -> Dict[int, List[int]]:
        neighbours: Dict[int, List[int]] = {a.id: [] for a in agents}
        alive = [a for a in agents if a.alive]
        if len(alive) < 2:
            return neighbours

        cell_size = max(self.cfg.comm_range, 1e-9)
        buckets: Dict[Tuple[int, int], List[Agent]] = defaultdict(list)
        for agent in alive:
            buckets[(int(agent.position[0] // cell_size), int(agent.position[1] // cell_size))].append(agent)

        r2 = self.cfg.comm_range * self.cfg.comm_range
        seen = set()
        for key, bucket_agents in buckets.items():
            kx, ky = key
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(buckets.get((kx + dx, ky + dy), []))
            for a in bucket_agents:
                for b in candidates:
                    if a.id >= b.id or (a.id, b.id) in seen:
                        continue
                    seen.add((a.id, b.id))
                    delta = a.position - b.position
                    if float(np.dot(delta, delta)) > r2:
                        continue
                    p1 = a.position.copy(); p1[2] += 0.5
                    p2 = b.position.copy(); p2[2] += 0.5
                    if terrain.line_of_sight(p1, p2):
                        neighbours[a.id].append(b.id)
                        neighbours[b.id].append(a.id)
        return neighbours

    def enqueue_messages(self, agents: List[Agent], neighbours: Dict[int, List[int]], t: int) -> None:
        """Broadcast each alive sender's current opinion and local quality observation."""
        agent_map = {a.id: a for a in agents}
        for sender_id, nbr_ids in neighbours.items():
            sender = agent_map[sender_id]
            if not sender.alive:
                continue
            for receiver_id in nbr_ids:
                receiver = agent_map[receiver_id]
                distance = float(np.linalg.norm(sender.position - receiver.position))
                opinion = self.apply_noise(sender.opinion, sender, receiver, t)
                if self.apply_packet_loss(sender, receiver, t):
                    continue
                quality = self.effective_quality(sender, receiver, distance, t)
                self._queue.append(Message(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    opinion=opinion,
                    quality=quality,
                    send_time=t,
                    deliver_time=self.apply_delay(t, distance),
                ))

    def deliver_messages(self, t: int) -> DeliveredMap:
        delivered: DeliveredMap = {}
        remaining: List[Message] = []
        for msg in self._queue:
            if msg.deliver_time <= t:
                delivered.setdefault(msg.receiver_id, []).append(msg)
            else:
                remaining.append(msg)
        self._queue = remaining
        return delivered

    def apply_delay(self, current_time: int, distance: float) -> int:
        if self.cfg.distance_delay_scale <= 0:
            distance_delay = 0
        else:
            distance_delay = int(math.ceil(distance / self.cfg.distance_delay_scale))
        jitter = int(self._rng.integers(0, self.cfg.delay_jitter_max + 1)) if self.cfg.delay_jitter_max > 0 else 0
        return current_time + self.cfg.base_delay + distance_delay + jitter

    def apply_noise(self, opinion: int, sender: Agent, receiver: Agent, t: int) -> int:
        if self.cfg.noise_flip_prob > 0.0 and self._rng.random() < self.cfg.noise_flip_prob:
            return 1 - opinion
        return opinion

    def apply_packet_loss(self, sender: Agent, receiver: Agent, t: int) -> bool:
        return self.cfg.drop_probability > 0.0 and self._rng.random() < self.cfg.drop_probability

    def effective_quality(self, sender: Agent, receiver: Agent, distance: float, t: int) -> float:
        base = sender.quality_for_current_opinion(self.option_cfg)
        ghost_decay = self._ghost_time_decay(sender, t)
        distance_decay = self._distance_decay(distance)
        return float(np.clip(base * ghost_decay * distance_decay, 0.0, 1.0))

    def _ghost_time_decay(self, sender: Agent, t: int) -> float:
        if sender.mobile or sender.stuck_since < 0:
            return 1.0
        window = max(self.failure_cfg.ghost_recognition_window, 1)
        return max(0.0, 1.0 - (max(t - sender.stuck_since, 0) / window))

    def _distance_decay(self, distance: float) -> float:
        if self.cfg.distance_decay_scale <= 0:
            return 1.0
        return float(math.exp(-distance / self.cfg.distance_decay_scale))
