# Experiment 2 calibration selection

Condition: beacon period 10 ticks, base delay 4, jitter cap 2, loss 0.10. Training seeds 101-115 (15 runs per candidate).
Rule (fixed in run_cache_experiment.py before running): maximise mean post-swap correct fraction; ties -> lower median recovery time.

| Policy | Candidates | Selected | Mean post-swap correct |
|---|---|---|---|
| Hard TTL | 10, 20, 40, 80 ticks | 20 | 0.806 |
| Fixed decay | 0.01, 0.02, 0.05, 0.10 per tick | 0.10 (upper edge of grid) | 0.810 |
| FreshVote | kappa 0.5, 1.0, 1.5, 2.0 | 0.5 | 0.797 |
| Baseline (reference) | - | - | 0.695 |

Differences among FreshVote candidates (0.789-0.797) and among the better TTL/decay candidates are within seed noise.
Calibration runs are training data only and are not pooled with the held-out evaluation (seeds 201-240).
Full candidate table: calibration_candidates.csv; run-level rows: calibration_runs.csv.
