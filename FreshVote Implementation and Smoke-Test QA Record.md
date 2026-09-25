# FreshVote Implementation and Smoke-Test QA Record

**Date:** 18 August 2026  
**Purpose:** Verify the isolated FreshVote implementation before calibration or publication-scale experimentation.

## Implementation status

FreshVote has been implemented in `/home/ubuntu/freshvote_project/freshvote/`, a separate workspace created from the user-supplied baseline source. The original extracted baseline remains unchanged. The implementation adds: retained message timestamps; non-negative receiver-side age calculation; bounded integer message jitter; an explicit best-of-two option model with locally noisy quality observations; baseline, hard-TTL, fixed-decay, and adaptive FreshVote policies; local rolling age histories; persistence-based correct/incorrect/transient consensus labels; and message acceptance/age counters.

The extension also corrects a validity limitation in the old model: the new study’s broadcast quality is tied to the option currently advertised, while the receiver uses only the sender’s local observation and timestamp. The true better option is known only to the evaluator for outcome scoring.

| Verification activity | Result | Interpretation |
|---|---:|---|
| Legacy and new unit tests | **40 passed** | Core terrain, communication, consensus, determinism, and five FreshVote-specific contracts passed. |
| Isolated smoke matrix | **64 completed runs** | Every policy executed under two delays, two jitter caps, two loss values, and two seeds. |
| Output integrity | CSV and JSON manifest written | Every smoke run contains outcome, freshness acceptance, rejected-age count, and stable-decision fields. |
| Environment record | Captured | Python 3.12.3, NumPy 2.5.1, pytest 9.1.1, and single-thread numerical settings were recorded. |

## Smoke-test result

The smoke matrix deliberately uses only 20 agents, 300 ticks, two seed pairs, and an easier 0.95/20-tick stability check. It is therefore **not publication evidence** and must not appear in the paper. Its function is to establish that the code operates end to end and that the mechanism counters react as expected.

| Policy | Runs | Stable-correct rate | Transient-threshold rate | Mean fresh-message acceptance | Age-rejected messages |
|---|---:|---:|---:|---:|---:|
| Baseline | 16 | 1.000 | 0.875 | 1.000 | 0 |
| Fixed decay | 16 | 1.000 | 1.000 | 1.000 | 0 |
| Hard TTL | 16 | 0.688 | 0.812 | 0.547 | 416,415 |
| Adaptive FreshVote | 16 | 1.000 | 1.000 | 0.957 | 40,145 |

The expected qualitative implementation checks are satisfied: the baseline and fixed-decay controls retain all messages; the hard TTL rejects a large volume of messages at delayed conditions; and adaptive FreshVote rejects a smaller volume of messages based on local age observations. All policy paths produce stable-decision and transient-threshold classifications.

The reported transient-threshold rate is intentionally broad: it flags *any* threshold visit that ends before a persistence window, including early transient visits followed by a later stable decision. It is a mechanism diagnostic, not a failure rate. The publication analysis will separately report stable correct consensus, stable incorrect consensus, non-decision, and post-hit loss using the frozen 0.99/300-tick definition.

> **Decision:** The implementation has passed unit and integration QA. The next valid stage is held-out parameter calibration at the pre-specified representative stress condition; no claim of FreshVote superiority has been made or is warranted from the smoke test.

## Reproducibility assets

| Asset | Location |
|---|---|
| Isolated implementation | `/home/ubuntu/freshvote_project/freshvote/code/` |
| Unit tests | `/home/ubuntu/freshvote_project/freshvote/code/tests/test_freshvote.py` |
| Smoke runner | `/home/ubuntu/freshvote_project/freshvote/code/run_freshvote_smoke.py` |
| Smoke raw summary | `/home/ubuntu/freshvote_project/freshvote/results/smoke/smoke_summary.csv` |
| Typed smoke aggregate | `/home/ubuntu/freshvote_project/freshvote/results/smoke/smoke_policy_summary.csv` |
| Environment record | `/home/ubuntu/freshvote_project/freshvote/manifests/environment_smoke_2026-08-18.txt` |
