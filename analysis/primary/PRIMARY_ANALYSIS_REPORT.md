# FreshVote primary held-out evaluation

The uploaded primary dataset passed all integrity checks: 3,840 unique runs, 96 factor cells, 40 held-out seeds per cell, exact frozen policy settings, and an execution log that reconciles one-to-one with the CSV. The analysis below uses no calibration or smoke data.

## Main finding

Every policy reached stable correct consensus in every run. The observed stable-correct rate is therefore 100% (960/960) for each policy; the two-sided Wilson 95% interval is 99.60%–100.00%. There were no stable-incorrect or non-decision outcomes. This is a complete ceiling, so the data do **not** support a claim that FreshVote improves decision correctness or completion probability over the baseline.

The defensible contribution is instead a boundary and trade-off result. Across the balanced matrix, median stable-decision onset was 24.10 s for the baseline, 24.10 s for hard TTL, 23.45 s for fixed decay, and 23.50 s for FreshVote. The pooled paired mean timing change for FreshVote versus baseline was -0.12 [-0.69, 0.44] s; its pooled paired median change was -0.20 [-0.60, 0.30] s. FreshVote was faster in 490 matched runs, tied in 8, and slower in 462. At the condition-median level it was faster in 13/24 cells, tied in 0/24, and slower in 11/24. This pattern is mixed rather than uniformly beneficial.

## Stress boundary

At the most severe tested cell (base delay 6 ticks, jitter cap 4 ticks, 10% packet loss), all policies still achieved 40/40 stable-correct decisions. Median onset times were 39.65 s (baseline), 62.45 s (hard TTL), 42.30 s (fixed decay), and 43.45 s (FreshVote). Hard TTL accepted only 34.5% of delivered messages in this cell, compared with 100.0% for FreshVote. The fixed TTL therefore over-filtered when absolute age rose, whereas the adaptive threshold expanded with the local age distribution. FreshVote avoided the hard-TTL slowdown but did not beat the unfiltered baseline in this most severe cell.

## Mechanism and diagnostic interpretation

Across all conditions, hard TTL rejected 11.1% of delivered messages, while FreshVote rejected 3.5%. Fixed decay and the baseline have 100% positive-weight acceptance by definition; fixed decay changes sampling weights rather than rejecting messages. Delivered-message counts were identical across matched policies, confirming that the intervention affected receiver-side decision weighting rather than network traffic.

The `transient_consensus` and `post_hit_loss` diagnostics are nearly saturated because they mark any early 99% threshold visit that ends before the required 300-tick persistence window, even if a later stable-correct decision occurs. They must not be interpreted as failures. All inferential timing comparisons use the onset of the eventual 300-tick stable-correct interval. No right-censoring procedure was activated because every run observed the event.

## Statistical scope

Stable-correct proportions use two-sided Wilson 95% intervals. Policy contrasts use deterministic non-parametric bootstrap resampling of matched seed pairs (20,000 replicates). Condition-specific intervals are descriptive and are not multiplicity-adjusted. The results establish behavior only for the simulated 100-agent regroup task, the tested delay/jitter/loss grid, the 0.15 option-quality gap, and the frozen parameter settings.

## Reproducibility record

Primary CSV SHA-256: `9950b5d9dc363d073a93a6ae1af5f66d6b1892a458fd0e321a5ee40743038362`.

The uploaded archive did not include the execution machine’s Python/NumPy versions or `pip freeze`. Those details should be recovered from the Linux environment before final manuscript submission.

## Generated artifacts

- `overall_policy_summary.csv`: overall outcomes, Wilson intervals, paired timing contrasts, and message filtering.
- `condition_summary.csv`: all 96 policy-condition cells.
- `paired_contrasts_vs_baseline.csv`: 72 matched intervention-versus-baseline contrasts.
- `delay_profiles.csv`: delay-stratified median timing profiles and bootstrap intervals.
- `fig_timing_contrast_heatmaps.*`: condition-level paired median timing changes.
- `fig_delay_profiles.*`: median stable-decision onset by delay and loss.
