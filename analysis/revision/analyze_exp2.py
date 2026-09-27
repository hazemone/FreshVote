"""Experiment 2 analysis: held-out neighbour-cache runs (seeds 201-240).

Outputs (analysis/exp2/): condition_summary.csv, overall_summary.csv, paired_contrasts.csv,
and figures in figures/. Paired contrasts resample seeds within condition.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent)); import style; style.apply()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/exp2"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
B = 20_000
DT = 0.1


def seed_of(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def boot(values, stat, key):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed_of(key))
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = v[idx]
    est = s.mean(1) if stat == "mean" else np.median(s, 1)
    return np.quantile(est, [0.025, 0.975])


def boot_grouped(df, col, key):
    """Paired bootstrap over seeds, stratified by condition (pooled estimate = mean)."""
    rng = np.random.default_rng(seed_of(key))
    groups = [g[col].to_numpy() for _, g in df.groupby("condition")]
    est = np.zeros(B)
    n = sum(len(g) for g in groups)
    for g in groups:
        idx = rng.integers(0, len(g), size=(B, len(g)))
        est += g[idx].sum(1)
    est /= n
    return np.quantile(est, [0.025, 0.975])


d = pd.read_csv(ROOT / "results/cache_experiment/evaluation_runs.csv")
d["channel"] = np.where(d.base_delay == 0, "benign", "severe")
d["condition"] = "B" + d.beacon_period.astype(str) + "-" + d.channel
d["rec_s"] = d.recovery_ticks * DT
d["rec_inf"] = d.rec_s.fillna(np.inf)
d["pre_s"] = d.pre_decision_tick * DT
d["pre_inf"] = d.pre_s.fillna(np.inf)  # censored: no 90% decision before the swap
assert len(d) == 960 and d.groupby(["policy", "condition"]).size().eq(40).all()

# ---- condition summary -------------------------------------------------------
rows = []
for (pol, cond), g in d.groupby(["policy", "condition"]):
    lo, hi = boot(g.post_mean_correct, "mean", f"post|{pol}|{cond}")
    rows.append(dict(policy=pol, condition=cond, beacon=g.beacon_period.iloc[0], channel=g.channel.iloc[0],
                     post_mean=g.post_mean_correct.mean(), post_lo=lo, post_hi=hi,
                     rec_median=np.median(g.rec_inf), not_recovered=int((~g.recovered).sum()),
                     pre_median=np.median(g.pre_inf), pre_missing=int(g.pre_s.isna().sum()),
                     acceptance=g.acceptance.mean(), hold=g.hold_share.mean(), stale=g.stale_share.mean(),
                     accepted_age_s=g.mean_accepted_age.mean() * DT))
cs = pd.DataFrame(rows); cs.to_csv(OUT / "condition_summary.csv", index=False)

# ---- paired contrasts ----------------------------------------------------------
key = ["condition", "seed"]
piv = {p: d[d.policy == p].set_index(key).sort_index() for p in style.POLICIES}
crow = []
for ref in ("baseline", "adaptive_freshvote"):
    for p in style.POLICIES:
        if p == ref:
            continue
        diff = (piv[p].post_mean_correct - piv[ref].post_mean_correct).reset_index()
        diff.columns = ["condition", "seed", "diff"]
        lo, hi = boot_grouped(diff, "diff", f"pooled|{p}|{ref}")
        crow.append(dict(policy=p, reference=ref, condition="pooled", diff=diff["diff"].mean(), lo=lo, hi=hi,
                         wins=int((diff["diff"] > 0).sum()), losses=int((diff["diff"] < 0).sum())))
        for cond, g in diff.groupby("condition"):
            lo, hi = boot(g["diff"], "mean", f"c|{p}|{ref}|{cond}")
            crow.append(dict(policy=p, reference=ref, condition=cond, diff=g["diff"].mean(), lo=lo, hi=hi,
                             wins=int((g["diff"] > 0).sum()), losses=int((g["diff"] < 0).sum())))
pc = pd.DataFrame(crow); pc.to_csv(OUT / "paired_contrasts.csv", index=False)

# recovery-time paired contrasts (censored runs set to +inf; median of paired differences)
rrow = []
for p in style.POLICIES[1:]:
    a, b = piv[p].rec_inf.to_numpy(), piv["baseline"].rec_inf.to_numpy()
    rrow.append(dict(policy=p, faster=int((a < b).sum()), slower=int((a > b).sum()), ties=int((a == b).sum())))
pd.DataFrame(rrow).to_csv(OUT / "recovery_sign_counts.csv", index=False)

# ---- overall summary -------------------------------------------------------------
orow = []
for p in style.POLICIES:
    g = d[d.policy == p]
    lo, hi = boot_grouped(g, "post_mean_correct", f"overall|{p}")
    orow.append(dict(policy=p, post_mean=g.post_mean_correct.mean(), post_lo=lo, post_hi=hi,
                     rec_median=np.median(g.rec_inf), not_recovered=int((~g.recovered).sum()),
                     pre_median=np.median(g.pre_inf), pre_missing=int(g.pre_s.isna().sum()),
                     acceptance=g.acceptance.mean(), hold=g.hold_share.mean()))
ov = pd.DataFrame(orow); ov.to_csv(OUT / "overall_summary.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print(ov.round(4)); print(pc.round(4)); print(pd.DataFrame(rrow))
print(cs[["policy", "condition", "post_mean", "post_lo", "post_hi", "rec_median", "not_recovered", "pre_median",
          "acceptance", "hold", "stale", "accepted_age_s"]].round(3).to_string())

# ---- figure: post-swap accuracy per condition --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
offs = dict(zip(style.POLICIES, np.linspace(-0.27, 0.27, 4)))
for ax, ch in zip(axes, ["benign", "severe"]):
    sub = cs[cs.channel == ch]
    for p in style.POLICIES:
        s = sub[sub.policy == p].sort_values("beacon")
        x = np.arange(3) + offs[p]
        ax.errorbar(x, s.post_mean, yerr=[s.post_mean - s.post_lo, s.post_hi - s.post_mean], fmt=style.MARKERS[p],
                    color=style.COLORS[p], ms=4, lw=1, capsize=0, label=style.LABELS[p])
    ax.set_xticks(range(3)); ax.set_xticklabels(["0.2", "1", "3"]); ax.set_xlabel("Beacon period $B$ (s)")
    ax.set_title("Benign channel" if ch == "benign" else "Severe channel")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Post-swap correct fraction")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.1), frameon=False, handletextpad=0.3, columnspacing=0.9)
fig.savefig(FIG / "fig_exp2_accuracy.pdf"); fig.savefig(FIG / "fig_exp2_accuracy.png", dpi=300)
print("done")
