#!/usr/bin/env python3
"""
Step 3 — Detection ROC and Extended Data Fig. 4 (panels a, b)
==============================================================

Builds the watermark-detection ROC for each post-processing method, separately
for protein–ligand and protein–antibody complexes, and reports TPR at a fixed
operating point (default FPR = 1e-5).

Detection statistic and null
----------------------------
The detector emits K bits.  The test statistic is the number of bits matching
the embedded message, k.  Under H0 (no watermark, or the wrong key) each bit is
an independent fair coin, so

    FPR(tau) = P[X >= tau],  X ~ Binomial(K, 0.5)

is available analytically — which is what makes an FPR as small as 1e-5
quotable without needing 1e5 negative structures.  That null is the one
validated by the per-bit calibration analysis (decoded bits on unwatermarked
structures are ~0.5 per position and near-uncorrelated).

TPR is **empirical**: the fraction of watermarked structures whose k reaches
the threshold,

    TPR(tau) = (1/N) * #{ structures with k >= tau }

so every point on the curve is measured, not modelled.  Because TPR is a step
function of an integer statistic, the curve is a staircase over tau = 0..K; the
reported TPR at a target FPR uses the smallest tau whose analytic FPR does not
exceed the target (the conservative choice).

Output
------
outputs/step3/
    roc_points.csv        every (dataset, method, tau, FPR, TPR)
    tpr_at_fpr.csv        TPR at the target FPR, per dataset and method
    relaxation_roc.png / .pdf

Run:
    python experiments_revision/robustness_postprocessing/step3_roc_and_figure.py \\
        --scores_csv experiments_revision/robustness_postprocessing/outputs/step2/detection_scores.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_ops import METHOD_LABELS  # noqa: E402

DEFAULT_SCORES = (
    Path(__file__).resolve().parent / "outputs" / "step2" / "detection_scores.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step3"

# Style matched to the other revision ROC figures.
METHOD_ORDER = ["none", "amber", "rosetta_fastrelax", "rosetta_ngk"]
COLORS  = ["silver", "darkorange", "darkcyan", "violet"]
MARKERS = ["o", "*", "p", "h"]
MARKER_SIZE = 5
LINE_WIDTH = 2

DATASET_TITLES = {
    "ligand": "Protein–ligand complexes",
    "antibody": "Protein–antibody complexes",
}


def analytic_fpr(K: int) -> tuple[np.ndarray, np.ndarray]:
    """tau = 0..K and the corresponding FPR = P[Binomial(K, 0.5) >= tau]."""
    from scipy.stats import binom

    tau = np.arange(0, K + 1)
    return tau, 1.0 - binom.cdf(tau - 1, K, 0.5)


def empirical_tpr(k_values: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Fraction of observed match counts reaching each threshold."""
    k_values = np.asarray(k_values)
    return np.array([(k_values >= t).mean() for t in tau])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3: detection ROC under post-processing (Extended Data Fig. 4 a, b)"
    )
    ap.add_argument("--scores_csv", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--target_fpr", type=float, default=1e-5)
    args = ap.parse_args()

    if not args.scores_csv.is_file():
        sys.exit(f"[error] Step 2 scores not found: {args.scores_csv}\n"
                 "        Run step2_detect_watermark.py first.")

    df = pd.read_csv(args.scores_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    K = int(df["n_bits"].mode().iat[0])
    if (df["n_bits"] != K).any():
        print(f"[step3] WARNING: mixed code widths in the scores file "
              f"{sorted(df['n_bits'].unique())}; using K={K} and dropping the rest.")
        df = df[df["n_bits"] == K]

    tau, fpr = analytic_fpr(K)
    min_fpr = float(fpr[-1])
    if args.target_fpr < min_fpr:
        print(f"[step3] WARNING: target FPR {args.target_fpr:g} is below the smallest\n"
              f"        FPR a {K}-bit code can express (2^-{K} = {min_fpr:.3g}).\n"
              f"        Reporting at {min_fpr:.3g} instead.")

    datasets = [d for d in ["ligand", "antibody"] if d in set(df["dataset"])]
    datasets += [d for d in sorted(set(df["dataset"])) if d not in datasets]

    roc_rows, op_rows = [], []
    for dataset in datasets:
        for method in METHOD_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if sub.empty:
                continue
            k_values = sub["n_matching_bits"].to_numpy()
            tpr = empirical_tpr(k_values, tau)

            for t, f, p in zip(tau, fpr, tpr):
                roc_rows.append({"dataset": dataset, "method": method,
                                 "tau": int(t), "fpr": float(f), "tpr": float(p)})

            usable = np.where(fpr <= max(args.target_fpr, min_fpr))[0]
            idx = int(usable[0]) if usable.size else len(tau) - 1
            op_rows.append({
                "dataset": dataset, "method": method,
                "method_label": METHOD_LABELS[method],
                "n_structures": int(len(sub)), "n_bits": K,
                "threshold_tau": int(tau[idx]),
                "achieved_fpr": float(fpr[idx]),
                "tpr": float(tpr[idx]),
                "mean_bit_accuracy": float(sub["bit_accuracy"].mean()),
                "sd_bit_accuracy": float(sub["bit_accuracy"].std(ddof=1))
                                   if len(sub) > 1 else float("nan"),
            })

    if not op_rows:
        sys.exit("[error] no (dataset, method) groups found in the scores file")

    roc = pd.DataFrame(roc_rows)
    ops = pd.DataFrame(op_rows)
    roc.to_csv(args.out_dir / "roc_points.csv", index=False)
    ops.to_csv(args.out_dir / "tpr_at_fpr.csv", index=False)

    print(f"[step3] K = {K} bits; TPR at FPR <= {args.target_fpr:g}:")
    print(ops[["dataset","method_label","n_structures","threshold_tau",
               "achieved_fpr","tpr","mean_bit_accuracy"]].to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.4),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        for method, color, marker in zip(METHOD_ORDER, COLORS, MARKERS):
            sub = roc[(roc["dataset"] == dataset) & (roc["method"] == method)]
            if sub.empty:
                continue
            # Drop tau = 0 (FPR = 1 by construction) for a readable log axis.
            sub = sub[sub["fpr"] > 0]
            ax.plot(sub["fpr"], sub["tpr"], label=METHOD_LABELS[method],
                    color=color, marker=marker, markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH)

        ax.set_xscale("log")
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=14)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=14)
        ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=14)
        ax.set_ylim(-0.03, 1.05)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=12, width=1.5)
        ax.yaxis.set_tick_params(labelsize=12, width=1.5)

    axes[0].legend(frameon=False, fontsize=11, loc="lower right")

    png = args.out_dir / "relaxation_roc.png"
    pdf = args.out_dir / "relaxation_roc.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[step3] ROC points → {args.out_dir/'roc_points.csv'}")
    print(f"[step3] Operating point → {args.out_dir/'tpr_at_fpr.csv'}")
    print(f"[step3] Figure → {png}")
    print("\n[step3] TPR is empirical over the scored structures; FPR is the\n"
          "        analytic Binomial(K, 0.5) tail. n per curve is in tpr_at_fpr.csv.\n")


if __name__ == "__main__":
    main()
