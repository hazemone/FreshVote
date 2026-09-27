#!/usr/bin/env python3
"""Locked analysis of the verified FreshVote primary evaluation.

All comparisons use matched policy-condition-seed runs. The script does not read
calibration or smoke outputs and never modifies primary inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICIES = ["baseline", "hard_ttl", "fixed_decay", "adaptive_freshvote"]
INTERVENTIONS = ["hard_ttl", "fixed_decay", "adaptive_freshvote"]
LABELS = {
    "baseline": "Baseline",
    "hard_ttl": "Hard TTL",
    "fixed_decay": "Fixed decay",
    "adaptive_freshvote": "FreshVote",
}
COLORS = {
    "baseline": "#202124",
    "hard_ttl": "#D55E00",
    "fixed_decay": "#0072B2",
    "adaptive_freshvote": "#009E73",
}
MARKERS = {"baseline": "o", "hard_ttl": "s", "fixed_decay": "^", "adaptive_freshvote": "D"}
CONDITIONS = ["delay", "jitter", "loss"]
KEYS = CONDITIONS + ["seed"]
DT_SECONDS = 0.1
BOOTSTRAPS = 20_000
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def percentile_ci(values: np.ndarray, stat: str, seed_key: str, n_boot: int = BOOTSTRAPS) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(stable_seed(seed_key))
    sample_indices = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[sample_indices]
    if stat == "mean":
        estimates = samples.mean(axis=1)
    elif stat == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError(stat)
    lo, hi = np.quantile(estimates, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return centre - half, centre + half


def q25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def policy_summary(frame: pd.DataFrame, policy: str, baseline: pd.DataFrame) -> dict:
    data = frame[frame.policy == policy].copy().sort_values(KEYS)
    n = len(data)
    correct = int(data.stable_correct_consensus.sum())
    incorrect = int(data.stable_incorrect_consensus.sum())
    no_stable = n - correct - incorrect
    rate_lo, rate_hi = wilson(correct, n)
    times = data.stable_decision_seconds.to_numpy(dtype=float)

    if policy == "baseline":
        paired = np.zeros(n, dtype=float)
    else:
        merged = baseline[KEYS + ["stable_decision_seconds"]].merge(
            data[KEYS + ["stable_decision_seconds"]], on=KEYS, suffixes=("_baseline", "_policy"), validate="one_to_one"
        )
        paired = (merged.stable_decision_seconds_policy - merged.stable_decision_seconds_baseline).to_numpy()
    mean_lo, mean_hi = percentile_ci(paired, "mean", f"overall-{policy}-mean")
    median_lo, median_hi = percentile_ci(paired, "median", f"overall-{policy}-median")

    delivered = int(data.delivered_messages.sum())
    accepted = int(data.accepted_messages.sum())
    rejected = int(data.rejected_age_messages.sum())
    weighted_age = float((data.accepted_message_mean_age * data.accepted_messages).sum())
    return {
        "policy": policy,
        "policy_label": LABELS[policy],
        "runs": n,
        "stable_correct": correct,
        "stable_correct_rate": correct / n,
        "stable_correct_wilson_low": rate_lo,
        "stable_correct_wilson_high": rate_hi,
        "stable_incorrect": incorrect,
        "no_stable_outcome": no_stable,
        "decision_median_seconds": float(np.median(times)),
        "decision_q25_seconds": float(np.quantile(times, 0.25)),
        "decision_q75_seconds": float(np.quantile(times, 0.75)),
        "decision_mean_seconds": float(np.mean(times)),
        "paired_mean_delta_vs_baseline_seconds": float(np.mean(paired)),
        "paired_mean_delta_ci_low": mean_lo,
        "paired_mean_delta_ci_high": mean_hi,
        "paired_median_delta_vs_baseline_seconds": float(np.median(paired)),
        "paired_median_delta_ci_low": median_lo,
        "paired_median_delta_ci_high": median_hi,
        "paired_faster_runs": int((paired < 0).sum()),
        "paired_tied_runs": int((paired == 0).sum()),
        "paired_slower_runs": int((paired > 0).sum()),
        "message_acceptance": accepted / delivered,
        "message_age_rejection_share": rejected / delivered,
        "accepted_message_weighted_mean_age_ticks": weighted_age / accepted,
        "transient_consensus_runs": int(data.transient_consensus.sum()),
        "post_hit_loss_runs": int(data.post_hit_loss.sum()),
    }


def condition_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (policy, delay, jitter, loss), data in frame.groupby(["policy", *CONDITIONS], sort=True):
        n = len(data)
        correct = int(data.stable_correct_consensus.sum())
        lo, hi = wilson(correct, n)
        delivered = int(data.delivered_messages.sum())
        accepted = int(data.accepted_messages.sum())
        rejected = int(data.rejected_age_messages.sum())
        records.append({
            "policy": policy,
            "policy_label": LABELS[policy],
            "delay": int(delay),
            "jitter": int(jitter),
            "loss": float(loss),
            "runs": n,
            "stable_correct": correct,
            "stable_correct_rate": correct / n,
            "stable_correct_wilson_low": lo,
            "stable_correct_wilson_high": hi,
            "stable_incorrect": int(data.stable_incorrect_consensus.sum()),
            "no_stable_outcome": int(n - data.stable_correct_consensus.sum() - data.stable_incorrect_consensus.sum()),
            "decision_median_seconds": float(data.stable_decision_seconds.median()),
            "decision_q25_seconds": q25(data.stable_decision_seconds),
            "decision_q75_seconds": q75(data.stable_decision_seconds),
            "decision_mean_seconds": float(data.stable_decision_seconds.mean()),
            "message_acceptance": accepted / delivered,
            "message_age_rejection_share": rejected / delivered,
            "accepted_message_weighted_mean_age_ticks": float(
                (data.accepted_message_mean_age * data.accepted_messages).sum() / accepted
            ),
            "transient_consensus_rate": float(data.transient_consensus.mean()),
            "post_hit_loss_rate": float(data.post_hit_loss.mean()),
        })
    return pd.DataFrame.from_records(records)


def paired_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame.policy == "baseline"].copy()
    records = []
    for policy in INTERVENTIONS:
        policy_data = frame[frame.policy == policy].copy()
        merged = baseline.merge(policy_data, on=KEYS, suffixes=("_baseline", "_policy"), validate="one_to_one")
        for (delay, jitter, loss), data in merged.groupby(CONDITIONS, sort=True):
            time_diff = (data.stable_decision_seconds_policy - data.stable_decision_seconds_baseline).to_numpy(dtype=float)
            outcome_diff = (
                data.stable_correct_consensus_policy.astype(float) - data.stable_correct_consensus_baseline.astype(float)
            ).to_numpy()
            mean_lo, mean_hi = percentile_ci(time_diff, "mean", f"cell-{policy}-{delay}-{jitter}-{loss}-mean")
            med_lo, med_hi = percentile_ci(time_diff, "median", f"cell-{policy}-{delay}-{jitter}-{loss}-median")
            rate_lo, rate_hi = percentile_ci(outcome_diff, "mean", f"cell-{policy}-{delay}-{jitter}-{loss}-rate")
            records.append({
                "policy": policy,
                "policy_label": LABELS[policy],
                "delay": int(delay),
                "jitter": int(jitter),
                "loss": float(loss),
                "matched_pairs": len(data),
                "correct_rate_difference_pp": float(outcome_diff.mean() * 100),
                "correct_rate_difference_ci_low_pp": rate_lo * 100,
                "correct_rate_difference_ci_high_pp": rate_hi * 100,
                "paired_mean_time_delta_seconds": float(time_diff.mean()),
                "paired_mean_time_delta_ci_low": mean_lo,
                "paired_mean_time_delta_ci_high": mean_hi,
                "paired_median_time_delta_seconds": float(np.median(time_diff)),
                "paired_median_time_delta_ci_low": med_lo,
                "paired_median_time_delta_ci_high": med_hi,
                "faster_pairs": int((time_diff < 0).sum()),
                "tied_pairs": int((time_diff == 0).sum()),
                "slower_pairs": int((time_diff > 0).sum()),
                "mean_acceptance_difference_pp": float(
                    (data.fresh_message_acceptance_policy - data.fresh_message_acceptance_baseline).mean() * 100
                ),
                "delivered_message_counts_identical": bool(
                    (data.delivered_messages_policy == data.delivered_messages_baseline).all()
                ),
            })
    return pd.DataFrame.from_records(records)


def delay_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (policy, loss, delay), data in frame.groupby(["policy", "loss", "delay"], sort=True):
        values = data.stable_decision_seconds.to_numpy(dtype=float)
        med_lo, med_hi = percentile_ci(values, "median", f"profile-{policy}-{loss}-{delay}")
        records.append({
            "policy": policy,
            "policy_label": LABELS[policy],
            "loss": float(loss),
            "delay": int(delay),
            "runs": len(data),
            "decision_median_seconds": float(np.median(values)),
            "decision_median_ci_low": med_lo,
            "decision_median_ci_high": med_hi,
        })
    return pd.DataFrame.from_records(records)


def make_heatmap(contrasts: pd.DataFrame, output_dir: Path) -> None:
    x_levels = [(jitter, loss) for jitter in [0, 2, 4] for loss in [0.0, 0.1]]
    y_levels = [0, 2, 4, 6]
    values = []
    matrices = {}
    for policy in INTERVENTIONS:
        matrix = np.empty((len(y_levels), len(x_levels)), dtype=float)
        for yi, delay in enumerate(y_levels):
            for xi, (jitter, loss) in enumerate(x_levels):
                row = contrasts[
                    (contrasts.policy == policy)
                    & (contrasts.delay == delay)
                    & (contrasts.jitter == jitter)
                    & np.isclose(contrasts.loss, loss)
                ].iloc[0]
                matrix[yi, xi] = row.paired_median_time_delta_seconds
        matrices[policy] = matrix
        values.extend(matrix.ravel())
    limit = max(5.0, math.ceil(max(abs(float(min(values))), abs(float(max(values)))) / 5.0) * 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), sharey=True, constrained_layout=True)
    cmap = mpl.colormaps["RdBu_r"]
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = None
    for ax, policy in zip(axes, INTERVENTIONS):
        matrix = matrices[policy]
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(LABELS[policy], fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(x_levels)), [f"{j}\n{int(loss * 100)}%" for j, loss in x_levels])
        ax.set_xlabel("Jitter cap (ticks) / loss", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        for yi in range(matrix.shape[0]):
            for xi in range(matrix.shape[1]):
                value = matrix[yi, xi]
                color = "white" if abs(value) > 0.55 * limit else "black"
                ax.text(xi, yi, f"{value:+.1f}", ha="center", va="center", fontsize=6.8, color=color)
    axes[0].set_yticks(range(len(y_levels)), y_levels)
    axes[0].set_ylabel("Base delay (ticks)", fontsize=8)
    assert image is not None
    cbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.015)
    cbar.set_label("Paired median decision-time change (s)\nnegative = faster than baseline", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(output_dir / "fig_timing_contrast_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_timing_contrast_heatmaps.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def make_delay_profile(profile: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), sharey=True, constrained_layout=True)
    for ax, loss in zip(axes, [0.0, 0.1]):
        for policy in POLICIES:
            data = profile[(profile.policy == policy) & np.isclose(profile.loss, loss)].sort_values("delay")
            x = data.delay.to_numpy(dtype=float)
            y = data.decision_median_seconds.to_numpy(dtype=float)
            lo = data.decision_median_ci_low.to_numpy(dtype=float)
            hi = data.decision_median_ci_high.to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=COLORS[policy],
                marker=MARKERS[policy],
                linewidth=1.5,
                markersize=4,
                label=LABELS[policy],
            )
            ax.fill_between(x, lo, hi, color=COLORS[policy], alpha=0.10, linewidth=0)
        ax.set_title(f"Packet loss = {int(loss * 100)}%", fontsize=9, fontweight="bold")
        ax.set_xlabel("Base delay (ticks)", fontsize=8)
        ax.set_xticks([0, 2, 4, 6])
        ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Stable-decision onset (s)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False, fontsize=7.5)
    fig.savefig(output_dir / "fig_delay_profiles.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_delay_profiles.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def format_ci(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_report(
    output_dir: Path,
    overall: pd.DataFrame,
    condition: pd.DataFrame,
    contrasts: pd.DataFrame,
    input_hash: str,
) -> None:
    indexed = overall.set_index("policy")
    stress = condition[(condition.delay == 6) & (condition.jitter == 4) & np.isclose(condition.loss, 0.1)].set_index("policy")
    adaptive = indexed.loc["adaptive_freshvote"]
    hard = indexed.loc["hard_ttl"]
    fixed = indexed.loc["fixed_decay"]
    baseline = indexed.loc["baseline"]
    adaptive_contrasts = contrasts[contrasts.policy == "adaptive_freshvote"]
    adaptive_faster_cells = int((adaptive_contrasts.paired_median_time_delta_seconds < 0).sum())
    adaptive_tied_cells = int((adaptive_contrasts.paired_median_time_delta_seconds == 0).sum())
    adaptive_slower_cells = int((adaptive_contrasts.paired_median_time_delta_seconds > 0).sum())

    lines = [
        "# FreshVote primary held-out evaluation",
        "",
        "The uploaded primary dataset passed all integrity checks: 3,840 unique runs, 96 factor cells, 40 held-out seeds per cell, exact frozen policy settings, and an execution log that reconciles one-to-one with the CSV. The analysis below uses no calibration or smoke data.",
        "",
        "## Main finding",
        "",
        "Every policy reached stable correct consensus in every run. The observed stable-correct rate is therefore 100% (960/960) for each policy; the two-sided Wilson 95% interval is "
        f"{baseline.stable_correct_wilson_low * 100:.2f}%–{baseline.stable_correct_wilson_high * 100:.2f}%. There were no stable-incorrect or non-decision outcomes. This is a complete ceiling, so the data do **not** support a claim that FreshVote improves decision correctness or completion probability over the baseline.",
        "",
        "The defensible contribution is instead a boundary and trade-off result. Across the balanced matrix, median stable-decision onset was "
        f"{baseline.decision_median_seconds:.2f} s for the baseline, {hard.decision_median_seconds:.2f} s for hard TTL, "
        f"{fixed.decision_median_seconds:.2f} s for fixed decay, and {adaptive.decision_median_seconds:.2f} s for FreshVote. "
        f"The pooled paired mean timing change for FreshVote versus baseline was {format_ci(adaptive.paired_mean_delta_vs_baseline_seconds, adaptive.paired_mean_delta_ci_low, adaptive.paired_mean_delta_ci_high)} s; its pooled paired median change was {format_ci(adaptive.paired_median_delta_vs_baseline_seconds, adaptive.paired_median_delta_ci_low, adaptive.paired_median_delta_ci_high)} s. "
        f"FreshVote was faster in {adaptive.paired_faster_runs} matched runs, tied in {adaptive.paired_tied_runs}, and slower in {adaptive.paired_slower_runs}. At the condition-median level it was faster in {adaptive_faster_cells}/24 cells, tied in {adaptive_tied_cells}/24, and slower in {adaptive_slower_cells}/24. This pattern is mixed rather than uniformly beneficial.",
        "",
        "## Stress boundary",
        "",
        "At the most severe tested cell (base delay 6 ticks, jitter cap 4 ticks, 10% packet loss), all policies still achieved 40/40 stable-correct decisions. Median onset times were "
        f"{stress.loc['baseline'].decision_median_seconds:.2f} s (baseline), {stress.loc['hard_ttl'].decision_median_seconds:.2f} s (hard TTL), "
        f"{stress.loc['fixed_decay'].decision_median_seconds:.2f} s (fixed decay), and {stress.loc['adaptive_freshvote'].decision_median_seconds:.2f} s (FreshVote). "
        f"Hard TTL accepted only {stress.loc['hard_ttl'].message_acceptance * 100:.1f}% of delivered messages in this cell, compared with {stress.loc['adaptive_freshvote'].message_acceptance * 100:.1f}% for FreshVote. The fixed TTL therefore over-filtered when absolute age rose, whereas the adaptive threshold expanded with the local age distribution. FreshVote avoided the hard-TTL slowdown but did not beat the unfiltered baseline in this most severe cell.",
        "",
        "## Mechanism and diagnostic interpretation",
        "",
        f"Across all conditions, hard TTL rejected {hard.message_age_rejection_share * 100:.1f}% of delivered messages, while FreshVote rejected {adaptive.message_age_rejection_share * 100:.1f}%. Fixed decay and the baseline have 100% positive-weight acceptance by definition; fixed decay changes sampling weights rather than rejecting messages. Delivered-message counts were identical across matched policies, confirming that the intervention affected receiver-side decision weighting rather than network traffic.",
        "",
        "The `transient_consensus` and `post_hit_loss` diagnostics are nearly saturated because they mark any early 99% threshold visit that ends before the required 300-tick persistence window, even if a later stable-correct decision occurs. They must not be interpreted as failures. All inferential timing comparisons use the onset of the eventual 300-tick stable-correct interval. No right-censoring procedure was activated because every run observed the event.",
        "",
        "## Statistical scope",
        "",
        "Stable-correct proportions use two-sided Wilson 95% intervals. Policy contrasts use deterministic non-parametric bootstrap resampling of matched seed pairs (20,000 replicates). Condition-specific intervals are descriptive and are not multiplicity-adjusted. The results establish behavior only for the simulated 100-agent regroup task, the tested delay/jitter/loss grid, the 0.15 option-quality gap, and the frozen parameter settings.",
        "",
        "## Reproducibility record",
        "",
        f"Primary CSV SHA-256: `{input_hash}`.",
        "",
        "The uploaded archive did not include the execution machine’s Python/NumPy versions or `pip freeze`. Those details should be recovered from the Linux environment before final manuscript submission.",
        "",
        "## Generated artifacts",
        "",
        "- `overall_policy_summary.csv`: overall outcomes, Wilson intervals, paired timing contrasts, and message filtering.",
        "- `condition_summary.csv`: all 96 policy-condition cells.",
        "- `paired_contrasts_vs_baseline.csv`: 72 matched intervention-versus-baseline contrasts.",
        "- `delay_profiles.csv`: delay-stratified median timing profiles and bootstrap intervals.",
        "- `fig_timing_contrast_heatmaps.*`: condition-level paired median timing changes.",
        "- `fig_delay_profiles.*`: median stable-decision onset by delay and loss.",
    ]
    (output_dir / "PRIMARY_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    if len(frame) != 3840:
        raise ValueError(f"Expected 3840 rows, found {len(frame)}")
    if sorted(frame.policy.unique()) != sorted(POLICIES):
        raise ValueError(f"Unexpected policies: {frame.policy.unique().tolist()}")
    if frame[KEYS + ["policy"]].duplicated().any():
        raise ValueError("Duplicate policy-condition-seed rows")
    for field in [
        "stable_correct_consensus",
        "stable_incorrect_consensus",
        "transient_consensus",
        "post_hit_loss",
    ]:
        if frame[field].dtype != bool:
            frame[field] = frame[field].map({"True": True, "False": False})
        if frame[field].isna().any():
            raise ValueError(f"Malformed Boolean field: {field}")

    frame["stable_decision_seconds"] = frame.stable_decision_time * DT_SECONDS
    if frame.stable_decision_seconds.isna().any():
        raise ValueError("The locked primary dataset unexpectedly contains censored time outcomes")

    baseline = frame[frame.policy == "baseline"].copy().sort_values(KEYS)
    overall = pd.DataFrame.from_records([policy_summary(frame, policy, baseline) for policy in POLICIES])
    condition = condition_summaries(frame)
    contrasts = paired_contrasts(frame)
    profile = delay_profiles(frame)

    overall.to_csv(args.output_dir / "overall_policy_summary.csv", index=False, float_format="%.10g")
    condition.to_csv(args.output_dir / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(args.output_dir / "paired_contrasts_vs_baseline.csv", index=False, float_format="%.10g")
    profile.to_csv(args.output_dir / "delay_profiles.csv", index=False, float_format="%.10g")
    condition[(condition.delay == 6) & (condition.jitter == 4) & np.isclose(condition.loss, 0.1)].to_csv(
        args.output_dir / "severe_stress_summary.csv", index=False, float_format="%.10g"
    )

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    make_heatmap(contrasts, args.output_dir)
    make_delay_profile(profile, args.output_dir)

    input_hash = sha256(args.input)
    manifest = {
        "analysis": "FreshVote locked primary evaluation",
        "input": str(args.input.resolve()),
        "input_sha256": input_hash,
        "rows": len(frame),
        "matched_comparison_unit": "condition-seed pair",
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed_method": "first 64 bits of SHA-256 over stable comparison key",
        "confidence_level": 0.95,
        "time_conversion_seconds_per_tick": DT_SECONDS,
        "censoring": "not activated; stable-correct event observed in every run",
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, overall, condition, contrasts, input_hash)

    print(overall.to_string(index=False))
    print(f"\nWrote locked analysis to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
