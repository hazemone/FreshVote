"""Metrics for agreement, decision correctness, stability, and message freshness."""

from typing import Dict, List, Optional
import numpy as np

from agent import Agent
from config import MetricsConfig


class MetricsRecorder:
    def __init__(self, cfg: MetricsConfig, best_option: Optional[int] = None):
        self.cfg = cfg
        self.best_option = best_option
        self.time: List[int] = []
        self.fraction_one: List[float] = []
        self.correct_fraction: List[float] = []
        self.order_param: List[float] = []
        self.variance: List[float] = []
        self.fiedler: List[float] = []
        self.alive_count: List[int] = []
        self.mobile_count: List[int] = []
        self.ghost_count: List[int] = []
        self.collapsed: List[bool] = []
        self.fragmented: List[bool] = []
        self.delivered_messages: List[int] = []
        self.accepted_messages: List[int] = []
        self.rejected_age_messages: List[int] = []
        self.accepted_age_mean: List[float] = []
        self._collapse_step: Optional[int] = None
        self._min_variance: float = float("inf")
        self._warmup_done: bool = False
        self._threshold_option: Optional[int] = None
        self._threshold_start: Optional[int] = None
        self._stable_outcome: Optional[str] = None
        self._stable_decision_time: Optional[int] = None
        self._transient_consensus: bool = False
        self._post_hit_loss: bool = False
        self._residence_ticks: int = 0

    def record(self, agents: List[Agent], neighbours: Dict[int, List[int]], t: int,
               decision_stats: Optional[dict] = None) -> None:
        if t % self.cfg.record_interval != 0:
            return
        alive = [a for a in agents if a.alive]
        mobile = [a for a in alive if a.mobile]
        ghosts = [a for a in alive if not a.mobile]
        opinions = np.array([a.opinion for a in alive], dtype=np.int8)
        if len(opinions) == 0:
            p_one, var, order = 0.0, 0.0, 0.0
        else:
            p_one = float(opinions.mean())
            var = p_one * (1.0 - p_one)
            order = abs(2.0 * p_one - 1.0)
        correct_fraction = (p_one if self.best_option == 1 else 1.0 - p_one) if self.best_option is not None else float("nan")

        if t % self.cfg.fiedler_interval == 0 or not self.fiedler:
            fiedler_val = self._compute_fiedler(alive, neighbours)
        else:
            fiedler_val = self.fiedler[-1]
        fragmented = fiedler_val < self.cfg.fiedler_fragment_threshold
        if not self._warmup_done:
            self._min_variance = min(self._min_variance, var)
            if self.variance and var < self.variance[0] * 0.5:
                self._warmup_done = True
        else:
            self._min_variance = min(self._min_variance, var)
        is_collapsed = self._warmup_done and var > self.cfg.collapse_variance_threshold
        if is_collapsed and self._collapse_step is None:
            self._collapse_step = t

        self._update_stability(p_one, len(alive), t)
        stats = decision_stats or {}
        self.time.append(t)
        self.fraction_one.append(p_one)
        self.correct_fraction.append(correct_fraction)
        self.order_param.append(order)
        self.variance.append(var)
        self.fiedler.append(fiedler_val)
        self.alive_count.append(len(alive))
        self.mobile_count.append(len(mobile))
        self.ghost_count.append(len(ghosts))
        self.collapsed.append(is_collapsed)
        self.fragmented.append(fragmented)
        self.delivered_messages.append(int(stats.get("delivered_messages", 0)))
        self.accepted_messages.append(int(stats.get("accepted_messages", 0)))
        self.rejected_age_messages.append(int(stats.get("rejected_age_messages", 0)))
        self.accepted_age_mean.append(float(stats.get("accepted_age_mean", 0.0)))

    def _update_stability(self, p_one: float, alive_count: int, t: int) -> None:
        if alive_count == 0:
            threshold_option = None
        elif p_one >= self.cfg.consensus_threshold:
            threshold_option = 1
        elif (1.0 - p_one) >= self.cfg.consensus_threshold:
            threshold_option = 0
        else:
            threshold_option = None

        if threshold_option is None:
            if self._threshold_option is not None and self._stable_outcome is None:
                self._transient_consensus = True
                self._post_hit_loss = True
            self._threshold_option = None
            self._threshold_start = None
            return

        self._residence_ticks += self.cfg.record_interval
        if self._threshold_option != threshold_option:
            if self._threshold_option is not None and self._stable_outcome is None:
                self._transient_consensus = True
                self._post_hit_loss = True
            self._threshold_option = threshold_option
            self._threshold_start = t

        if self._stable_outcome is None and self._threshold_start is not None:
            if t - self._threshold_start + self.cfg.record_interval >= self.cfg.persistence_ticks:
                if self.best_option is None:
                    self._stable_outcome = "stable"
                elif threshold_option == self.best_option:
                    self._stable_outcome = "correct"
                else:
                    self._stable_outcome = "incorrect"
                self._stable_decision_time = self._threshold_start

    def finalize(self) -> None:
        """Classify an unfinished threshold visit as transient at end of run."""
        if self._threshold_option is not None and self._stable_outcome is None:
            self._transient_consensus = True

    def _compute_fiedler(self, alive: List[Agent], neighbours: Dict[int, List[int]]) -> float:
        if len(alive) < 2:
            return 0.0
        id_to_idx = {a.id: i for i, a in enumerate(alive)}
        laplacian = np.zeros((len(alive), len(alive)), dtype=np.float64)
        for agent in alive:
            i = id_to_idx[agent.id]
            for neighbour_id in neighbours.get(agent.id, []):
                if neighbour_id in id_to_idx:
                    j = id_to_idx[neighbour_id]
                    laplacian[i, j] = -1.0
                    laplacian[i, i] += 1.0
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
        return float(max(eigenvalues[1], 0.0)) if len(eigenvalues) >= 2 else 0.0

    @property
    def collapse_timestep(self) -> Optional[int]:
        return self._collapse_step

    def summary(self) -> dict:
        delivered = int(sum(self.delivered_messages))
        accepted = int(sum(self.accepted_messages))
        rejected = int(sum(self.rejected_age_messages))
        weighted_age_total = float(sum(a * n for a, n in zip(self.accepted_age_mean, self.accepted_messages)))
        return {
            "total_steps_recorded": len(self.time),
            "final_fraction_one": self.fraction_one[-1] if self.fraction_one else None,
            "final_correct_fraction": self.correct_fraction[-1] if self.correct_fraction else None,
            "final_order_param": self.order_param[-1] if self.order_param else None,
            "final_variance": self.variance[-1] if self.variance else None,
            "final_fiedler": self.fiedler[-1] if self.fiedler else None,
            "final_alive": self.alive_count[-1] if self.alive_count else None,
            "final_mobile": self.mobile_count[-1] if self.mobile_count else None,
            "final_ghost": self.ghost_count[-1] if self.ghost_count else None,
            "collapse_detected": self._collapse_step is not None,
            "collapse_timestep": self._collapse_step,
            "final_fragmented": self.fragmented[-1] if self.fragmented else None,
            "stable_outcome": self._stable_outcome,
            "stable_correct_consensus": self._stable_outcome == "correct",
            "stable_incorrect_consensus": self._stable_outcome == "incorrect",
            "stable_decision_time": self._stable_decision_time,
            "transient_consensus": self._transient_consensus,
            "post_hit_loss": self._post_hit_loss,
            "consensus_residence_ticks": self._residence_ticks,
            "delivered_messages": delivered,
            "accepted_messages": accepted,
            "rejected_age_messages": rejected,
            "fresh_message_acceptance": accepted / delivered if delivered else 0.0,
            "accepted_message_mean_age": weighted_age_total / accepted if accepted else 0.0,
        }
