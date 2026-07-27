#!/usr/bin/env python3
"""
Step 3 — Statistics and Extended Data Fig. 1 (panels b, c)
===========================================================

Reads the paired per-sequence scores from Step 2 and produces:

  1. Two-sided **Wilcoxon signed-rank** tests, per sequence length, for
     · watermark bit recovery accuracy   (FoldMark vs Chen et al.)
     · structural quality, scTM          (FoldMark vs Chen et al.)
     The test is paired because Step 1 generated each arm's sequence *i* from
     the same seed and the same message.

  2. Holm–Bonferroni correction across the tested lengths, reported alongside
     the raw p-values.  The stars drawn on the figure use the **corrected**
     p-values.

  3. The two-panel bar figure (Extended Data Fig. 1, panels b and c): mean ±
     s.d., significance bars, Arial, thick left/bottom spines.

Every number in the figure comes from the Step 2 CSV — nothing is hard-coded.

Output : outputs/step3/
           statistics.csv                        ← n, medians, W, p, p_holm, stars
           watermark_comparison_chen_vs_foldmark.png / .pdf

Run:
    python experiments_revision/chen_et_al_comparison/step3_statistics_and_figure.py \\
        --paired_csv experiments_revision/chen_et_al_comparison/outputs/step2/paired_scores.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PAIRED = Path(__file__).resolve().parent / "outputs" / "step2" / "paired_scores.csv"
OUT_DIR        = Path(__file__).resolve().parent / "outputs" / "step3"

COLORS = {"chen": "silver", "foldmark": "#5CA8D1"}
BAR_WIDTH = 0.36


def stars_for(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down correction, order preserved."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(np.where(finite, p, np.inf))
    m = int(finite.sum())
    running = 0.0
    for rank, i in enumerate(idx[:m]):
        adj = (m - rank) * p[i]
        running = max(running, adj)
        out[i] = min(running, 1.0)
    return out.tolist()


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """
    Two-sided Wilcoxon signed-rank on paired samples (b - a).
    Returns (statistic, p-value, n_pairs_used).  Pairs with NaN are dropped;
    if every difference is zero the test is undefined and NaN is returned.
    """
    from scipy.stats import wilcoxon

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(mask.sum())
    if n < 1 or np.allclose(a, b):
        return float("nan"), float("nan"), n
    stat, p = wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p), n


def _style_axes(ax) -> None:
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=14, width=1.5)
    ax.yaxis.set_tick_params(labelsize=14, width=1.5)


def _draw_sig(ax, x_left: float, x_right: float, y: float, label: str) -> None:
    ax.plot([x_left, x_right], [y, y], color="black", linewidth=1.0)
    ax.text(0.5 * (x_left + x_right), y, label, ha="center", va="bottom", fontsize=11)


def _panel(ax, x, lengths, chen_mean, chen_sd, fm_mean, fm_sd, stats_labels,
           ylabel: str, percent: bool, zoom: bool = True) -> None:
    import matplotlib.pyplot as plt

    ax.bar(x - BAR_WIDTH / 2, chen_mean, BAR_WIDTH, yerr=chen_sd, capsize=0,
           color=COLORS["chen"], edgecolor="0.3", linewidth=1.0,
           error_kw=dict(linewidth=1.0, color="0.2"), label="Chen et al.")
    ax.bar(x + BAR_WIDTH / 2, fm_mean, BAR_WIDTH, yerr=fm_sd, capsize=0,
           color=COLORS["foldmark"], edgecolor="0.2", linewidth=1.0,
           error_kw=dict(linewidth=1.0, color="0.2"), label="FoldMark")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in lengths])
    ax.set_xlabel("Sequence length (aa)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style_axes(ax)

    tops = np.maximum(np.asarray(chen_mean) + np.asarray(chen_sd),
                      np.asarray(fm_mean) + np.asarray(fm_sd))
    bots = np.minimum(np.asarray(chen_mean) - np.asarray(chen_sd),
                      np.asarray(fm_mean) - np.asarray(fm_sd))

    # Axis window. `zoom` matches the house style of the other revision figures
    # (bars do not start at zero); pass zoom=False for a zero-baseline axis.
    lo = max(0.0, float(np.nanmin(bots)) - 0.10) if zoom else 0.0
    span = float(np.nanmax(tops)) - lo or 1.0
    for i in range(len(lengths)):
        _draw_sig(ax, x[i] - BAR_WIDTH / 2, x[i] + BAR_WIDTH / 2,
                  tops[i] + 0.04 * span, stats_labels[i])
    ax.set_ylim(bottom=lo, top=float(np.nanmax(tops)) + 0.20 * span)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3: Wilcoxon signed-rank statistics and Extended Data Fig. 1 (b, c)"
    )
    ap.add_argument("--paired_csv", type=Path, default=DEFAULT_PAIRED)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--zero_baseline", action="store_true",
                    help="start both y-axes at 0 instead of zooming to the data range")
    args = ap.parse_args()

    if not args.paired_csv.is_file():
        raise FileNotFoundError(
            f"Step 2 paired scores not found: {args.paired_csv}\n"
            "Run step2_fold_and_score.py first."
        )
    df = pd.read_csv(args.paired_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted(df["length"].unique())
    print(f"[step3] {len(df)} pairs across lengths {lengths}")

    rows: list[dict] = []
    for metric, chen_col, fm_col in [
        ("bit_accuracy", "bit_accuracy_chen", "bit_accuracy_foldmark"),
        ("sctm",         "sctm_chen",         "sctm_foldmark"),
    ]:
        for L in lengths:
            sub = df[df["length"] == L]
            a = sub[chen_col].to_numpy(dtype=float)
            b = sub[fm_col].to_numpy(dtype=float)
            stat, p, n = wilcoxon_paired(a, b)
            rows.append({
                "metric": metric, "length": L, "n_pairs": n,
                "chen_mean": np.nanmean(a),   "chen_sd": np.nanstd(a, ddof=1),
                "chen_median": np.nanmedian(a),
                "foldmark_mean": np.nanmean(b), "foldmark_sd": np.nanstd(b, ddof=1),
                "foldmark_median": np.nanmedian(b),
                "median_difference": np.nanmedian(b) - np.nanmedian(a),
                "wilcoxon_W": stat, "p_value": p,
            })

    stats = pd.DataFrame(rows)
    # Holm correction within each metric family.
    stats["p_holm"] = np.nan
    for metric in stats["metric"].unique():
        m = stats["metric"] == metric
        stats.loc[m, "p_holm"] = holm(stats.loc[m, "p_value"].tolist())
    stats["stars"] = [stars_for(p) for p in stats["p_holm"]]
    stats["test"] = "two-sided Wilcoxon signed-rank (paired)"

    stats_path = args.out_dir / "statistics.csv"
    stats.to_csv(stats_path, index=False)
    print(f"[step3] Statistics → {stats_path}")
    print(stats[["metric","length","n_pairs","chen_mean","foldmark_mean",
                 "p_value","p_holm","stars"]].to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "Arial"
    x = np.arange(len(lengths))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    bit = stats[stats["metric"] == "bit_accuracy"].set_index("length").loc[lengths]
    _panel(ax0, x, lengths,
           bit["chen_mean"].to_numpy(), bit["chen_sd"].to_numpy(),
           bit["foldmark_mean"].to_numpy(), bit["foldmark_sd"].to_numpy(),
           bit["stars"].tolist(),
           ylabel="Watermark bit recovery accuracy", percent=True,
           zoom=not args.zero_baseline)

    sc = stats[stats["metric"] == "sctm"].set_index("length").loc[lengths]
    _panel(ax1, x, lengths,
           sc["chen_mean"].to_numpy(), sc["chen_sd"].to_numpy(),
           sc["foldmark_mean"].to_numpy(), sc["foldmark_sd"].to_numpy(),
           sc["stars"].tolist(),
           ylabel="scTM", percent=False, zoom=not args.zero_baseline)

    # Legend above the axes so it can never overlap a bar or a significance bar.
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=12,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))

    png = args.out_dir / "watermark_comparison_chen_vs_foldmark.png"
    pdf = args.out_dir / "watermark_comparison_chen_vs_foldmark.pdf"
    # bbox_inches="tight" so the legend placed above the axes is not clipped.
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[step3] Figure → {png}\n[step3] Figure → {pdf}")
    print(
        "\n[step3] Bars are mean ± s.d.; stars use Holm-corrected p-values from\n"
        "        two-sided Wilcoxon signed-rank tests on matched pairs.\n"
    )


if __name__ == "__main__":
    main()
