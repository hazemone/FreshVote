"""Experiment 1 figure: median stable-decision onset vs base delay (column width)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent)); import style; style.apply()
root = Path(__file__).resolve().parents[2]
p = pd.read_csv(root / "analysis/primary/delay_profiles.csv")
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
for ax, loss in zip(axes, [0, 0.1]):
    sub = p[np.isclose(p.loss, loss)]
    for pol in style.POLICIES:
        s = sub[sub.policy == pol].sort_values("delay")
        ax.fill_between(s.delay, s.decision_median_ci_low, s.decision_median_ci_high, color=style.COLORS[pol], alpha=0.12, lw=0)
        ax.plot(s.delay, s.decision_median_seconds, ls=style.LINES[pol], marker=style.MARKERS[pol], color=style.COLORS[pol], label=style.LABELS[pol])
    ax.set_title(f"Loss {int(loss*100)}%"); ax.set_xticks([0, 2, 4, 6]); ax.set_xlabel("Base delay $d_0$ (ticks)")
axes[0].set_ylabel("Decision onset (s)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False, handlelength=1.8, columnspacing=0.8)
out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "fig_exp1_delay.pdf"); fig.savefig(out / "fig_exp1_delay.png", dpi=300)
