"""Configuration for the FreshVote binary swarm-decision simulator."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TerrainConfig:
    width: int = 200
    depth: int = 400
    cell_size: float = 1.0
    seed: int = 42
    zone_count: int = 4
    min_density: float = 0.001
    max_density: float = 0.008


@dataclass
class AgentConfig:
    num_agents: int = 100
    base_speed: float = 0.5
    spawn_margin: float = 10.0
    initial_opinion_one_prob: float = 0.5
    quality_min: float = 0.5
    quality_max: float = 1.0


@dataclass
class OptionConfig:
    """Ground-truth best-of-two task, used only when explicitly enabled."""
    enabled: bool = False
    quality_zero: float = 0.60
    quality_one: float = 0.75
    measurement_noise_halfwidth: float = 0.05

    @property
    def best_option(self) -> int:
        if self.quality_zero == self.quality_one:
            raise ValueError("The two option qualities must differ for a correctness study.")
        return int(self.quality_one > self.quality_zero)


@dataclass
class ScenarioConfig:
    mode: str = "regroup"
    mission_heading: float = np.pi / 2
    heading_noise: float = 0.05
    repulsion_radius: float = 15.0
    repulsion_strength: float = 0.4
    obstacle_avoidance_radius: float = 3.0
    obstacle_avoidance_strength: float = 0.6


@dataclass
class CommsConfig:
    comm_range: float = 25.0
    base_delay: int = 0
    distance_delay_scale: float = 10.0
    delay_jitter_max: int = 0
    noise_flip_prob: float = 0.0
    drop_probability: float = 0.0
    distance_decay_scale: float = 25.0


@dataclass
class ConsensusConfig:
    update_probability: float = 1.0


@dataclass
class FreshnessConfig:
    """Receiver-side policy; the default preserves the legacy decision rule."""
    policy: str = "baseline"  # baseline | hard_ttl | fixed_decay | adaptive_freshvote
    fixed_ttl: int = 5
    fixed_decay_lambda: float = 0.20
    rolling_window_ticks: int = 50
    adaptive_kappa: float = 1.0
    min_ttl: int = 2
    cold_start_observations: int = 3
    clock_skew_max_ticks: int = 0


@dataclass
class FailureConfig:
    enabled: bool = True
    immobilise_probability: float = 0.0
    ghost_recognition_window: int = 50


@dataclass
class MetricsConfig:
    collapse_variance_threshold: float = 0.20
    record_interval: int = 1
    fiedler_interval: int = 25
    fiedler_fragment_threshold: float = 0.01
    consensus_threshold: float = 0.99
    persistence_ticks: int = 300


@dataclass
class SimConfig:
    dt: float = 0.1
    total_steps: int = 5000
    output_dir: str = "../data"
    random_seed: int = 42
    deployment_seed: int = 42


@dataclass
class Config:
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    option: OptionConfig = field(default_factory=OptionConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    sim: SimConfig = field(default_factory=SimConfig)
