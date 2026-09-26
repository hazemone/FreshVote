# FreshVote Protocol Specification and Experiment Manifest

**Version:** 1.0 — 18 August 2026  
**Status:** Protocol locked pending author approval; no FreshVote experiment has yet been run.

## 1. Objective and separation from the under-review manuscript

This study evaluates whether a **locally computed, timestamp-based freshness policy** can improve stable and correct collective decisions in a mobile robot swarm subject to delayed communication. The contribution is an intervention: FreshVote modifies how a receiver uses *already delivered* messages. It does not alter the terrain, movement model, neighbour discovery, range, line-of-sight test, failure model, or underlying message transport.

The study is deliberately distinct from the user’s under-review manuscript. That manuscript diagnoses delay-induced metastability using the existing consensus model. This study will introduce a new protocol, compare it to three explicit policies in a new decision-accuracy model, use new stability and communication metrics, and generate an entirely new dataset. No prior results or graphical material will be transferred into the workshop paper.

## 2. Decision environment

The study will use a binary best-of-two decision task with an objective better option. Let option \(1\) be the true best option, with \(Q_1=0.75\), and option \(0\) be inferior, with \(Q_0=0.60\). These values provide a non-trivial but visible quality gap. They are model choices, not external measurements, and will be stated exactly in the paper.

Each robot holds an opinion \(o_i\in\{0,1\}\). When robot \(i\) broadcasts opinion \(o_i\), its message contains a local quality observation

\[
\hat Q_{i,o_i}=\operatorname{clip}(Q_{o_i}+\epsilon_{i,o_i},\,0,\,1),
\]

where \(\epsilon_{i,o}\) is a robot–option-specific, zero-mean perturbation drawn once at deployment and retained throughout the run. The primary setting uses \(\epsilon_{i,o}\sim\mathcal{U}[-0.05,0.05]\). This gives each robot a local, imperfect but time-consistent assessment without allowing a receiver to access the true option quality. Initial opinions are independent fair draws, so the initial state does not favour the best option.

Every delivery retains the original sender timestamp. A receiver computes age as \(a_m=t_{\mathrm{receive}}-t_{\mathrm{send}}\). Timestamp error is zero in the primary study; a later sensitivity check uses a receiver-side clock offset sampled uniformly from \([-1,1]\) ticks.

## 3. Policies

All policies use the same quality-weighted, single-message-copy update family. They differ **only** in the receiver-side freshness multiplier applied to messages that arrive in the current tick. If no message has positive weight, a robot retains its current opinion for that tick.

| Policy | Selection weight for message \(m\) | Role |
|---|---|---|
| **Baseline** | \(w_m=\hat Q_m\) | Existing delayed weighted-voter control. |
| **Hard TTL** | \(w_m=\hat Q_m\mathbf{1}[a_m\leq\Theta]\) | Simple stale-message rejection control. |
| **Fixed decay** | \(w_m=\hat Q_m\exp(-\lambda a_m)\) | Non-adaptive age-discount control. |
| **FreshVote** | \(w_m=\hat Q_m\exp(-\lambda_i a_m)\mathbf{1}[a_m\leq\Theta_i]\) | Proposed locally adaptive method. |

For FreshVote, receiver \(i\) maintains a rolling window \(A_i\) of the ages of messages that it has received in the preceding \(W=50\) ticks. At each decision update,

\[
\Theta_i=\max(2,\operatorname{median}(A_i)+\kappa\operatorname{IQR}(A_i)), \qquad
\lambda_i=\frac{\ln 2}{\max(\operatorname{median}(A_i),1)}.
\]

During the cold start, before three age observations exist, FreshVote accepts all delivered messages with a provisional \(\lambda_i=0\) and \(\Theta_i=\infty\). This avoids an unjustified early bias against messages merely because a local age history has not yet formed. The sole adaptive-policy calibration parameter is \(\kappa\); all other values are fixed above. The policy requires only received timestamps and local storage, not centralised coordination or global delay knowledge.

## 4. Communication stress model

The baseline physical communication model will preserve range, line-of-sight, distance-dependent delay, and terrain/mobility exactly as in the existing simulator. The new timing stressor adds a non-negative per-message jitter term:

\[
\tau_m=\tau_0+\left\lceil d_m/10\right\rceil+J_m, \qquad J_m\sim\mathcal{U}\{0,\ldots,J_{\max}\}.
\]

The PRNG for jitter will be a dedicated stream. Packet loss remains an independent Bernoulli channel applied before queuing. All policies in a matched seed condition will share the same terrain seed, deployment seed, and initial communication schedule parameters. Their trajectories are then allowed to diverge through their legitimately different decisions.

## 5. Primary experiment

The primary experiment uses \(N=100\) robots, the `regroup` mobility scenario, a 200 m × 400 m obstacle field, 25 m communication range, 3000 ticks at 0.1 s/tick, and terrain/mobility settings frozen from the simulator configuration. The ghost-recognition model remains enabled as it is part of the stated stale-information setting.

| Factor | Levels | Rationale |
|---|---|---|
| Policy | Baseline, Hard TTL, Fixed decay, FreshVote | Tests the proposed adaptive mechanism against no freshness control and two simpler controls. |
| Base delay \(\tau_0\) | 0, 2, 4, 6 ticks | Covers no added base latency through the problematic delayed region without falsely treating a single threshold as universal. |
| Jitter cap \(J_{\max}\) | 0, 2, 4 ticks | Tests whether policy performance depends on timing variability rather than mean delay alone. |
| Packet-loss probability | 0, 0.10 | Tests a modest loss condition without conflating severe network collapse with freshness control. |
| Independent seed pairs | 40 per cell | Provides 40 paired terrain/deployment initialisations for every policy–stress cell. |

The full evaluation matrix contains \(4\times4\times3\times2\times40=3,840\) simulation runs. It will be executed in batches only after code-level smoke tests pass. All runs will be regenerated within one pinned environment; archived results are not inputs to this study.

## 6. Calibration and hold-out rule

Fixed controls must not be handicapped by arbitrary parameter choices, and FreshVote must not be tuned on its final test set. A separate training set of 20 seed pairs will be used only for calibration under the representative stress cell \((\tau_0,J_{\max},p_{\mathrm{loss}})=(4,2,0.10)\).

| Policy | Candidate settings | Selection rule |
|---|---|---|
| Hard TTL | \(\Theta\in\{3,5,7,9\}\) ticks | Maximise stable-correct-consensus rate; break ties by lower median stable-decision time. |
| Fixed decay | \(\lambda\in\{0.10,0.20,0.35,0.50\}\) tick\(^{-1}\) | Same rule. |
| FreshVote | \(\kappa\in\{0.5,1.0,1.5,2.0\}\) | Same rule. |

The selected values are then frozen and used for the 3,840 evaluation runs. Training seeds must not appear in the evaluation seed list. The paper will report the selected settings and state that tuning was held out.

## 7. Outcome definitions

A **stable consensus** is defined as a fraction of alive robots holding one option of at least \(0.99\) for \(H=300\) consecutive ticks. The 30-second persistence window prevents a one-tick snapshot from being mistaken for a reliable decision. If the alive population is empty, the run is classified as a failure, not a consensus.

| Outcome | Operational definition | Status |
|---|---|---|
| Stable correct consensus | Best-option fraction \(\ge0.99\) for \(H\) consecutive ticks. | **Primary outcome.** |
| Stable incorrect consensus | Inferior-option fraction \(\ge0.99\) for \(H\) consecutive ticks. | Safety outcome. |
| Transient consensus | Either option reaches 0.99 but never satisfies the \(H\)-tick persistence criterion. | Direct metastability outcome. |
| Non-decision | No stable consensus by the 3000-tick budget. | Cost-of-filtering outcome. |
| Stable-decision time | First tick at the beginning of the stable-correct persistence window; right-censored at 3000. | Efficiency outcome. |
| Post-hit loss rate | Proportion of threshold-hit runs that leave 0.99 before satisfying \(H\). | Stability outcome. |
| Fresh-message acceptance | Accepted received messages divided by delivered received messages. | Communication-use outcome. |
| Accepted-message age | Per-run median and mean age of messages given positive selection weight. | Mechanism check. |

Each rate will be reported with Wilson 95% confidence intervals. For matched-policy comparisons, the analysis will estimate the paired difference in stable-correct-consensus rate with a non-parametric bootstrap over seed pairs and a 95% confidence interval. Time-to-decision will be displayed as a right-censored empirical distribution or a restricted mean time to stable decision, depending on the observed censoring rate.

## 8. Planned figures and six-page budget

The main paper should contain at most four visual elements:

| Element | Content | Purpose |
|---|---|---|
| Fig. 1 | FreshVote message lifecycle and local adaptive age rule. | Makes the protocol reproducible. |
| Fig. 2 | Stable correct-consensus rate versus base delay, one panel per jitter/loss condition or a compact heatmap difference from baseline. | Primary effectiveness result. |
| Fig. 3 | Transient-consensus/post-hit loss rate and stable-decision time for the representative stress condition. | Establishes the claimed stabilisation mechanism and cost. |
| Table I | Policy comparison: stable correct, stable incorrect, median decision time, acceptance fraction, median accepted age. | Provides the stability–accuracy–communication trade-off. |

The final paper title, abstract, and claims will be selected only after the data are available. A negative or mixed outcome will not be discarded; it will be reported as an identified boundary of age-based freshness control.

## 9. Implementation verification checklist

Before the full experiment, the implementation must pass all of the following checks.

1. **Baseline equivalence:** With `policy=baseline`, zero jitter, and the original option-quality compatibility mode, the refactored delivery representation preserves existing result arrays for the same pinned stack.
2. **Age correctness:** Every delivered message obeys `age == receive_tick - send_time`; a base delay of zero still includes the existing distance component.
3. **Policy invariants:** Hard TTL rejects exactly messages above its threshold; fixed decay is monotone in age; FreshVote never uses future ages or global state; all-stale inboxes cause a hold action.
4. **Outcome correctness:** Synthetic trajectories verify correct, incorrect, transient, censored, and empty-swarm labels.
5. **Determinism:** Same configuration and seed pair are bit-identical within the pinned environment.
6. **No data leakage:** Calibration and evaluation seeds are disjoint; the final scripts record parameter values and git commit/hash in each run directory.
7. **Analysis integrity:** Analysis regenerates every table and figure from raw result files, with no manually entered result values.

## 10. Reproducibility record

The implementation will be developed in a clean `freshvote/` namespace that leaves the user-supplied baseline files unchanged. Before experiments, the team will create a pinned `requirements.txt` or lockfile, record Python/NumPy/package versions, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`, and write a machine-readable manifest for every run. The study will describe reproducibility as deterministic **within a pinned environment**, not as cross-environment bit-identical.

## 11. Approval gate

After author approval of this manifest, implementation will begin in the isolated FreshVote workspace. The first execution will be a small smoke-test matrix and unit tests, not the full experiment. No manuscript result, performance claim, or paper figure will be drafted before those checks pass.
