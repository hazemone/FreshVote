"""
fit_decay_curve.py
------------------
Operationalises the 'non-linear' claim in the report title.

For regroup N=100, fits two candidate models to consensus_rate(delay) over
delays {0, 1, 2, 3, 4, 5}:
  M1 (linear):       p(d) = a - b*d        (clipped to [0, 1])
  M2 (exponential):  p(d) = a * exp(-b*d)

Reports:
  - parameters
  - sum of squared errors
  - AIC (Akaike Information Criterion) for each
  - which model the data prefers and by how much

Outputs:
  - new_data/figures/curve_fit.png
  - prints fit summary to stdout
"""

from __future__ import annotations
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Pull the per-condition consensus rates we just computed.
# (mode=regroup, N=100, delays 0..5) — hard-coded from wilson_consensus_rates.csv
DELAYS = np.array([0, 1, 2, 3, 4, 5, 10, 20, 30, 50], dtype=float)
RATES = np.array([10/12, 7/12, 4/12, 3/12, 1/12, 0/12,
                  0/12, 0/12, 0/12, 0/12], dtype=float)
N_SEEDS = 12
N_OBS = len(DELAYS)


def linear_model(d, a, b):
    return np.clip(a - b * d, 0.0, 1.0)


def exp_model(d, a, b):
    return a * np.exp(-b * d)


def fit_linear(d, y):
    """Closed-form least-squares for unconstrained line a - b*d."""
    A = np.column_stack([np.ones_like(d), -d])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a, b


def fit_exp(d, y):
    """
    Grid search + local refinement for a*exp(-b*d), since y has zeros that
    rule out a log-linear fit.
    """
    best = (None, None, np.inf)
    for a_try in np.linspace(0.5, 1.0, 51):
        for b_try in np.linspace(0.1, 3.0, 91):
            yhat = exp_model(d, a_try, b_try)
            sse = float(np.sum((y - yhat) ** 2))
            if sse < best[2]:
                best = (a_try, b_try, sse)
    a0, b0, _ = best
    # local refinement
    for _ in range(3):
        a_grid = np.linspace(max(0.0, a0 - 0.05), min(1.0, a0 + 0.05), 41)
        b_grid = np.linspace(max(0.0, b0 - 0.2), b0 + 0.2, 41)
        for a_try in a_grid:
            for b_try in b_grid:
                yhat = exp_model(d, a_try, b_try)
                sse = float(np.sum((y - yhat) ** 2))
                if sse < best[2]:
                    best = (a_try, b_try, sse)
        a0, b0, _ = best
    return best  # (a, b, sse)


def aic(sse, n, k):
    """Gaussian-residual AIC: n*ln(SSE/n) + 2k."""
    if sse <= 0:
        sse = 1e-12
    return n * math.log(sse / n) + 2 * k


def main():
    a_lin, b_lin = fit_linear(DELAYS, RATES)
    sse_lin = float(np.sum((RATES - linear_model(DELAYS, a_lin, b_lin)) ** 2))
    aic_lin = aic(sse_lin, N_OBS, k=2)

    a_exp, b_exp, sse_exp = fit_exp(DELAYS, RATES)
    aic_exp = aic(sse_exp, N_OBS, k=2)

    print("=== Curve fit: regroup N=100, delays 0..5 ===")
    print(f"Observations: {list(zip(DELAYS.astype(int).tolist(), RATES.round(4).tolist()))}")
    print()
    print(f"Linear:       p(d) = {a_lin:.4f} - {b_lin:.4f} * d   "
          f"(clipped to [0,1])")
    print(f"  SSE = {sse_lin:.5f}   AIC = {aic_lin:.3f}")
    print()
    print(f"Exponential:  p(d) = {a_exp:.4f} * exp(-{b_exp:.4f} * d)")
    print(f"  SSE = {sse_exp:.5f}   AIC = {aic_exp:.3f}")
    print()
    delta = aic_lin - aic_exp
    if delta > 0:
        print(f"Exponential preferred over linear by delta_AIC = {delta:.3f}.")
    else:
        print(f"Linear preferred over exponential by delta_AIC = {-delta:.3f}.")
    print("(delta_AIC > 2 is conventionally 'positive' evidence; > 6 'strong'; > 10 'decisive'.)")

    dd = np.linspace(0, 50, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(DELAYS, RATES, "o", color="black", markersize=8,
            label="Observed (N=100, n=12)")
    ax.plot(dd, linear_model(dd, a_lin, b_lin), "--",
            color="#3377cc", linewidth=2,
            label=f"Linear fit (AIC={aic_lin:.1f})")
    ax.plot(dd, exp_model(dd, a_exp, b_exp), "-",
            color="#cc3322", linewidth=2,
            label=f"Exponential fit (AIC={aic_exp:.1f})")
    ax.set_xlabel("Base delay (ticks)")
    ax.set_ylabel("Consensus rate")
    ax.set_title("Regroup N=100: linear vs exponential decay of consensus rate (delay 0–50)")
    ax.set_xlim(-1, 51)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = ROOT / "new_data" / "figures" / "curve_fit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[fit] wrote {out}")


if __name__ == "__main__":
    main()
