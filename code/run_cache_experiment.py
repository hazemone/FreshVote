"""Experiment 2 runner: calibration (training seeds) and held-out evaluation.

Protocol (fixed before any calibration or evaluation run):
  * design pilots used seeds 1-4 only and are not reported as evidence;
  * calibration: condition B=10 ticks, (d0, J, p) = (4, 2, 0.10), seeds 101-115;
    selection = highest mean post-swap correct fraction, ties -> lower median recovery;
  * evaluation: B in {2, 10, 30} x channel in {benign (0,0,0), severe (6,4,0.10)},
    seeds 201-240, all four policies on identical seeds (common channel/motion streams).

Usage:
  python run_cache_experiment.py calibrate --workers 32
  python run_cache_experiment.py evaluate --workers 32 --ttl 20 --lam 0.05 --kappa 1.5
"""

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sim_cache import CacheConfig, run_cache

COMMON = dict(steps=6000, swap_tick=3000, update_probability=0.1, switch_probability=0.01,
              retention=600, threshold=0.90, persistence=100)
CAL_CONDITION = dict(beacon_period=10, base_delay=4, jitter=2, loss=0.10)
CAL_SEEDS = range(101, 116)
CANDIDATES = {
    "hard_ttl": ("ttl", [10, 20, 40, 80]),
    "fixed_decay": ("lam", [0.01, 0.02, 0.05, 0.10]),
    "adaptive_freshvote": ("kappa", [0.5, 1.0, 1.5, 2.0]),
}
EVAL_SEEDS = range(201, 241)
BEACONS = (2, 10, 30)
CHANNELS = {"benign": dict(base_delay=0, jitter=0, loss=0.0),
            "severe": dict(base_delay=6, jitter=4, loss=0.10)}
DROP = ("series_t", "series_c")


def _run(kw):
    out = run_cache(CacheConfig(**kw))
    for k in DROP:
        out.pop(k, None)
    return out


def _execute(jobs, workers, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_run, jobs, chunksize=1))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path} in {time.time() - t0:.0f} s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "evaluate"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ttl", type=float)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results" / "cache_experiment")
    a = ap.parse_args()

    if a.stage == "calibrate":
        jobs = [dict(COMMON, **CAL_CONDITION, policy="baseline", seed=s) for s in CAL_SEEDS]
        for pol, (name, values) in CANDIDATES.items():
            for v in values:
                jobs += [dict(COMMON, **CAL_CONDITION, policy=pol, **{name: v}, seed=s) for s in CAL_SEEDS]
        _execute(jobs, a.workers, a.out / "calibration_runs.csv")
        return

    frozen = dict(ttl=a.ttl, lam=a.lam, kappa=a.kappa)
    if None in frozen.values():
        raise SystemExit("evaluation requires the frozen --ttl, --lam and --kappa")
    jobs = []
    for pol in ("baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"):
        for b in BEACONS:
            for ch, chan in CHANNELS.items():
                for s in EVAL_SEEDS:
                    jobs.append(dict(COMMON, **chan, beacon_period=b, policy=pol, seed=s, **frozen))
    (a.out).mkdir(parents=True, exist_ok=True)
    (a.out / "evaluation_manifest.json").write_text(json.dumps(dict(
        common=COMMON, beacons=BEACONS, channels=CHANNELS, seeds=[EVAL_SEEDS.start, EVAL_SEEDS.stop - 1],
        frozen=frozen, calibration_condition=CAL_CONDITION,
        calibration_seeds=[CAL_SEEDS.start, CAL_SEEDS.stop - 1]), indent=2))
    _execute(jobs, a.workers, a.out / "evaluation_runs.csv")


if __name__ == "__main__":
    main()
