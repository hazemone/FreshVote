# FreshVote Held-Out Calibration Selection

**Date:** 19 August 2026  
**Training condition:** base delay 4 ticks, jitter cap 2 ticks, packet-loss probability 0.10, 20 independent seed pairs.  
**Selection rule:** maximise stable-correct-consensus rate; when tied, select the lower median stable-decision time among successful runs.  
**Evaluation seeds:** must be disjoint from training seeds 1–20.

| Policy | Candidates | Selected setting | Stable-correct rate | Median stable-decision time (ticks) | Selection basis |
|---|---|---:|---:|---:|---|
| Hard TTL | \(\Theta\in\{3,5,7,9\}\) | \(\Theta=9\) | 1.000 | 315 | The candidates at 7 and 9 both reached a rate of 1.000; 9 had the lower median decision time (315 vs. 329). |
| Fixed decay | \(\lambda\in\{0.10,0.20,0.35,0.50\}\) | \(\lambda=0.35\) | 1.000 | 262 | All candidates reached a rate of 1.000; 0.35 had the lowest median decision time. |
| FreshVote | \(\kappa\in\{0.5,1.0,1.5,2.0\}\) | \(\kappa=1.5\) | 1.000 | 270 | All candidates reached a rate of 1.000; 1.5 had the lowest median decision time. |

The calibration dataset contains **260 training-only simulation runs**. It is not part of the primary evaluation and must not be pooled with or selectively reported as test evidence. These parameter values are now frozen for the 3,840-run primary matrix specified in the approved protocol.

## Source artefacts

| Artefact | Path |
|---|---|
| Candidate summary | `results/calibration_full/calibration_candidates.csv` |
| Run-level training output | `results/calibration_full/calibration_runs.csv` |
| Calibration runner | `code/run_freshvote_calibration.py` |
