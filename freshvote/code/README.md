# Binary Weighted-Voter Swarm Consensus under Communication Delay

Code and data for an undergraduate research project on swarm consensus.

Title: *Communication Delay Causes Non-Linear Collapse in Decentralized
Swarm Consensus.*

## What is in the project

| Path | Purpose |
|---|---|
| `code/` | Python simulation source (this directory) |
| `code/tests/test_core.py` | Unit tests (pytest) |
| `code/run_sweep.py` | CPU-parallel sweep runner |
| `code/cliff_sweep.py` | Targeted cliff-edge sweep (delay 1–4) |
| `code/compute_wilson.py` | Aggregates sweep CSVs and attaches Wilson 95% CIs |
| `code/make_headline_figure.py` | Generates the consensus-rate-vs-delay figure |
| `code/fit_decay_curve.py` | Linear vs exponential fit + AIC for the cliff curve |
| `code/demo.py` | Live visualisation of a single run |
| `new_data/runs/` | Per-run `.npz` time-series files |
| `new_data/configs/` | Exact config JSON for each run |
| `new_data/summaries/` | Batch CSV summaries (incl. Wilson CIs) |
| `new_data/figures/` | Headline figure, zoom, and curve fit |
| `new_data/logs/` | Sweep timing logs |
| `assignment_model_notes.md` | Technical reference for the model |
| `results_claims.md` | Final aggregate results |
| `report_paragraphs.md` | Drop-in paragraphs for the report |

## Requirements

- Python 3.12 (any 3.10+ should work)
- Packages listed in `code/requirements.txt`

```powershell
pip install -r code\requirements.txt
```

## Running the simulation

Single live run (with visualisation):

```powershell
python code\demo.py
```

Headless single run (with metric save):

```powershell
python code\sim.py
```

## Running the full sweep

```powershell
python code\run_sweep.py --batch-index 0 --batch-size 120 --workers 8
```

The full delay sweep produces `2 modes × 3 swarm sizes × 10 delays × 12 seeds = 720`
runs across 6 batches. The cliff sweep adds 288 runs covering delays 1–4.

## Reproducing the figures

After running the sweep:

```powershell
python code\compute_wilson.py
python code\make_headline_figure.py
python code\fit_decay_curve.py
```

Outputs land under `new_data/summaries/` and `new_data/figures/`.

## Running the unit tests

From the `code` directory:

```powershell
pytest tests
```

## Key parameters

| Parameter | Default | Source |
|---|---|---|
| Swarm sizes | 25, 50, 100 | `run_sweep.SWARM_SIZES` |
| Base delays | 0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200 | sweep CLI |
| Seeds per condition | 12 | `run_sweep.seed_pairs` |
| Steps per run | 3000 | `run_sweep.run_job` |
| Communication range | 25 m | `config.CommsConfig.comm_range` |
| Distance delay scale | 10 m / extra tick | `config.CommsConfig.distance_delay_scale` |
| Distance decay scale | 25 m | `config.CommsConfig.distance_decay_scale` |
| Ghost recognition window | 50 ticks | `config.FailureConfig` |
| Obstacle density | 0.001–0.008 | `config.TerrainConfig` |
| Order parameter | `|2p − 1|` | `metrics.MetricsRecorder.record` |

## Reproducibility

Each run has two seeds: `terrain_seed` (obstacle layout) and
`deployment_seed` (agent positions, initial opinions, base qualities, and
all stochastic substreams). The deterministic seed pairs are constructed
in `run_sweep.seed_pairs`.

The same seed pair produces bit-identical metrics (verified in
`tests/test_core.py::test_simulation_same_seed_same_result`).
