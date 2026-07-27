#!/usr/bin/env python3
"""
Step 3 — Null calibration and the corrected tracing false-positive rate
========================================================================

The tracing formula is only as good as its null model.  The usual assumption is
that bits decoded from an **unwatermarked** structure behave as independent
Bernoulli(0.5) draws.  That is an assumption about a *learned* decoder, not a
theorem, and it can fail in ways that matter: the decoder may be biased at some
positions, its bits may be correlated with each other, or its behaviour may
depend on protein length, fold class, or prediction confidence.

This step measures all four, then recomputes the false-positive rate **without**
relying on the idealised null:

  a. **Per-position bias** — P(bit = 1) at each of the 32 positions, with an
     exact binomial test against 0.5 and Holm correction across positions.
  b. **Between-position dependence** — the pairwise correlation matrix, its
     largest off-diagonal magnitude, and a permutation test on that maximum.
  c. **Covariate dependence** — mean bit rate stratified by length bin, fold
     class and confidence bin, each with a Kruskal–Wallis test.
  d. **Calibrated FPR** — the probability that an unwatermarked structure is
     attributed to some user, computed three ways:
       · idealised    Binomial(32, 0.5) tail
       · Poisson-binomial using the measured per-position p_i
       · empirical    the observed rate over the supplied unwatermarked decodes
     Divergence between these three quantifies how much the idealised null
     misstates the real false-positive rate.

Input
-----
CSV of decoded bits from unwatermarked structures:

    structure_id,bits,length,fold_class,confidence
    1abc_A,01101001...,157,alpha,0.87

`bits` is a 32-character 0/1 string.  `length`, `fold_class` and `confidence`
are optional; the strata that are present get tested.

Output
------
outputs/step3/
    per_position_bias.csv
    position_correlations.csv
    covariate_stratification.csv
    fpr_calibration.csv
    null_calibration.png / .pdf

Run:
    python experiments_revision/user_tracing_codebook/step3_null_calibration.py \\
        --null_bits_csv unwatermarked_decodes.csv \\
        --codebook experiments_revision/user_tracing_codebook/outputs/step1/codebook.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step3"
DEFAULT_CODEBOOK = (
    Path(__file__).resolve().parent / "outputs" / "step1" / "codebook.npy"
)


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    idx = np.argsort(np.where(finite, p, np.inf))
    m = int(finite.sum())
    running = 0.0
    for rank, i in enumerate(idx[:m]):
        running = max(running, (m - rank) * p[i])
        out[i] = min(running, 1.0)
    return out.tolist()


def parse_bits(series: pd.Series, n_bits: int) -> np.ndarray:
    rows = []
    for s in series:
        b = [int(c) for c in str(s).strip() if c in "01"]
        if len(b) < n_bits:
            raise ValueError(f"bit string shorter than {n_bits}: {s!r}")
        rows.append(b[:n_bits])
    return np.array(rows, dtype=np.int8)


def poisson_binomial_tail(p: np.ndarray, k_min: int) -> float:
    """P[sum of independent Bernoulli(p_i) >= k_min], exact by convolution."""
    dist = np.array([1.0])
    for pi in p:
        dist = np.convolve(dist, [1.0 - pi, pi])
    return float(dist[k_min:].sum())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3: null calibration for the tracing false-positive rate"
    )
    ap.add_argument("--null_bits_csv", type=Path, required=True)
    ap.add_argument("--codebook",      type=Path, default=DEFAULT_CODEBOOK)
    ap.add_argument("--out_dir",       type=Path, default=OUT_DIR)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--n_permutations", type=int, default=1000)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    from scipy.stats import binomtest, kruskal

    if not args.null_bits_csv.is_file():
        sys.exit(f"[error] null bits CSV not found: {args.null_bits_csv}")
    df = pd.read_csv(args.null_bits_csv)
    if "bits" not in df.columns:
        sys.exit("[error] the CSV needs a 'bits' column of 0/1 strings")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    K = cb.CODE_N
    bits = parse_bits(df["bits"], K)
    n = bits.shape[0]
    rng = np.random.default_rng(args.seed)
    print(f"[step3] {n:,} unwatermarked decodes x {K} bits")

    # ── a. per-position bias ─────────────────────────────────────────────────
    ones = bits.sum(axis=0)
    p_hat = ones / n
    pvals = [binomtest(int(ones[i]), n, 0.5, alternative="two-sided").pvalue
             for i in range(K)]
    p_holm = holm(pvals)
    bias = pd.DataFrame({
        "position": np.arange(K),
        "n": n,
        "n_ones": ones,
        "p_hat": p_hat,
        "p_value": pvals,
        "p_holm": p_holm,
        "significant_holm_0.05": [p < 0.05 for p in p_holm],
    })
    bias.to_csv(args.out_dir / "per_position_bias.csv", index=False)
    n_biased = int(bias["significant_holm_0.05"].sum())
    print(f"[step3] a. per-position bias: mean p_hat = {p_hat.mean():.4f} "
          f"(range {p_hat.min():.4f}–{p_hat.max():.4f}); "
          f"{n_biased}/{K} positions differ from 0.5 (Holm, alpha=0.05)")

    # ── b. between-position dependence ───────────────────────────────────────
    x = bits.astype(float)
    sd = x.std(axis=0)
    const = sd == 0
    corr = np.full((K, K), np.nan)
    if (~const).sum() > 1:
        corr_sub = np.corrcoef(x[:, ~const], rowvar=False)
        idx = np.flatnonzero(~const)
        corr[np.ix_(idx, idx)] = corr_sub
    off = ~np.eye(K, dtype=bool)
    finite_off = np.isfinite(corr) & off
    max_abs = float(np.nanmax(np.abs(corr[finite_off]))) if finite_off.any() else float("nan")
    mean_abs = float(np.nanmean(np.abs(corr[finite_off]))) if finite_off.any() else float("nan")

    # Permutation test: shuffle each column independently to destroy any
    # between-position dependence while preserving per-position bias.
    perm_max = []
    for _ in range(args.n_permutations):
        xp = np.column_stack([rng.permutation(x[:, j]) for j in range(K)])
        sdp = xp.std(axis=0)
        keep = sdp > 0
        if keep.sum() < 2:
            continue
        cp = np.corrcoef(xp[:, keep], rowvar=False)
        m = ~np.eye(int(keep.sum()), dtype=bool)
        perm_max.append(np.abs(cp[m]).max())
    perm_p = (float(np.mean(np.array(perm_max) >= max_abs))
              if perm_max and np.isfinite(max_abs) else float("nan"))

    pd.DataFrame(corr).to_csv(args.out_dir / "position_correlations.csv", index=False)
    print(f"[step3] b. dependence: max |r| = {max_abs:.4f}, mean |r| = {mean_abs:.4f}, "
          f"permutation p = {perm_p:.4f}")

    # ── c. covariate dependence ──────────────────────────────────────────────
    per_structure_rate = bits.mean(axis=1)
    strat_rows = []
    for col, kind in [("length", "numeric"), ("confidence", "numeric"),
                      ("fold_class", "categorical")]:
        if col not in df.columns or df[col].isna().all():
            continue
        if kind == "numeric":
            try:
                groups_key = pd.qcut(df[col], q=4, duplicates="drop")
            except ValueError:
                continue
        else:
            groups_key = df[col].astype(str)

        buckets = [per_structure_rate[np.asarray(groups_key == g)]
                   for g in pd.unique(groups_key) if (groups_key == g).sum() > 0]
        buckets = [b for b in buckets if b.size > 0]
        stat, pv = (kruskal(*buckets) if len(buckets) > 1
                    else (float("nan"), float("nan")))
        for g in pd.unique(groups_key):
            m = np.asarray(groups_key == g)
            strat_rows.append({
                "covariate": col, "stratum": str(g), "n": int(m.sum()),
                "mean_bit_rate": float(per_structure_rate[m].mean()),
                "sd_bit_rate": float(per_structure_rate[m].std(ddof=1))
                               if m.sum() > 1 else float("nan"),
                "kruskal_H": float(stat), "kruskal_p": float(pv),
            })
        print(f"[step3] c. {col}: Kruskal–Wallis H = {stat:.3f}, p = {pv:.4g}")

    if strat_rows:
        pd.DataFrame(strat_rows).to_csv(
            args.out_dir / "covariate_stratification.csv", index=False)
    else:
        print("[step3] c. no covariate columns present — skipping stratification")

    # ── d. calibrated false-positive rate ────────────────────────────────────
    print("[step3] d. false-positive rate under three null models …")
    fpr_rows = []

    # The attribution rule accepts when the decoded word lies within `radius` of
    # a codeword, i.e. when at least K - radius bits match that codeword.
    k_min = K - args.radius

    from scipy.stats import binom as _binom
    ideal = float(1.0 - _binom.cdf(k_min - 1, K, 0.5))
    fpr_rows.append({
        "null_model": "idealised Binomial(K, 0.5)",
        "per_codeword_fpr": ideal,
        "note": "assumes unbiased, independent null bits",
    })

    pb = poisson_binomial_tail(p_hat, k_min)
    fpr_rows.append({
        "null_model": "Poisson-binomial with measured per-position p_i",
        "per_codeword_fpr": pb,
        "note": "relaxes the unbiasedness assumption; still assumes independence",
    })

    empirical = float("nan")
    if args.codebook.is_file():
        messages = np.load(args.codebook).astype(np.uint32)
        pool, _ = cb.build_pool()
        words = pool[messages]
        print(f"[step3]    scoring {n:,} null decodes against "
              f"{words.size:,} user codewords …")
        n_attr = 0
        for row in bits:
            q = np.uint32(int("".join(str(int(b)) for b in row[::-1]), 2))
            dist = cb.popcount(words ^ q)
            i1 = int(np.argmin(dist))
            d1 = int(dist[i1])
            saved = dist[i1]; dist[i1] = np.iinfo(np.int64).max
            d2 = int(dist.min()); dist[i1] = saved
            if d1 <= args.radius and (d2 - d1) >= args.margin:
                n_attr += 1
        empirical = n_attr / n
        fpr_rows.append({
            "null_model": "empirical over supplied unwatermarked decodes",
            "per_codeword_fpr": empirical,
            "note": f"whole-codebook attribution rate, {n_attr}/{n} attributed, "
                    f"radius={args.radius}, margin={args.margin}",
        })
    else:
        print(f"[step3]    codebook not found at {args.codebook}; "
              "skipping the empirical arm")

    fpr = pd.DataFrame(fpr_rows)
    fpr.to_csv(args.out_dir / "fpr_calibration.csv", index=False)
    print(fpr.to_string(index=False))

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax0.axhspan(0.45, 0.55, color="0.9", zorder=0)
    ax0.axhline(0.5, color="0.4", linestyle="--", linewidth=1.2, zorder=1)
    ax0.bar(np.arange(K), p_hat, color="#5CA8D1", edgecolor="0.2",
            linewidth=0.8, zorder=2)
    ax0.set_xlabel("Bit position", fontsize=13)
    ax0.set_ylabel("P(bit = 1) on unwatermarked structures", fontsize=13)
    ax0.set_ylim(0, 1)
    ax0.set_title(f"Per-position null bias (n = {n:,})", fontsize=13)

    im = ax1.imshow(np.where(np.isfinite(corr), corr, 0.0),
                    cmap="coolwarm", vmin=-0.3, vmax=0.3)
    ax1.set_xlabel("Bit position", fontsize=13)
    ax1.set_ylabel("Bit position", fontsize=13)
    ax1.set_title(f"Between-position correlation (max |r| = {max_abs:.3f})",
                  fontsize=13)
    fig.colorbar(im, ax=ax1, fraction=0.046, label="Pearson r")

    for ax in (ax0, ax1):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    png = args.out_dir / "null_calibration.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / "null_calibration.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\n[step3] Figure → {png}")
    print(f"[step3] Written to {args.out_dir}\n")
    if np.isfinite(empirical) and np.isfinite(ideal) and ideal > 0:
        print(f"[step3] Empirical / idealised FPR ratio: {empirical/ideal:.2f}x — "
              f"report this rather than the idealised value if it departs from 1.\n")


if __name__ == "__main__":
    main()
