#!/usr/bin/env python3
"""
Step 3 — ESM2 Ranking and Construct Selection for eGFP
=======================================================

Scores every MPNN-designed sequence (from Step 2) with the ESM2-650M
masked pseudo-log-likelihood (PLL), ranks them, and writes the top
constructs for wet-lab synthesis.

Selection criteria (as used in the manuscript):
    • sequence_recovery ≥ 80 %
    • n_mutations ≤ 25
    • ESM2-650M PLL ≥ wild-type PLL   (sequences at least as "natural" as WT)

Input  : tutorials/egfp/outputs/step2/mpnn_sequences.csv
Output : tutorials/egfp/outputs/step3/
           ranked_sequences.csv       ← all sequences with ESM2 scores + rank
           top_constructs.fasta       ← top-12 sequences for synthesis

Run from the foldmark_space root:
    python tutorials/egfp/step3_esm2_ranking.py

Dependencies:
    pip install fair-esm torch numpy pandas
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── eGFP wild-type sequence (same as Step 1; used for WT PLL baseline) ───────
EGFP_SEQUENCE = (
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWP"
    "TLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDT"
    "LVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQL"
    "ADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)

# Construct-selection thresholds
MIN_RECOVERY  = 0.80   # ≥ 80% overall sequence recovery
MAX_MUTATIONS = 25     # ≤ 25 amino-acid changes vs. wild-type
N_EXPORT      = 12     # constructs to export for wet-lab

DEFAULT_CSV = Path("tutorials/egfp/outputs/step2/mpnn_sequences.csv")
OUT_DIR     = Path("tutorials/egfp/outputs/step3")


# ── ESM2 scoring ──────────────────────────────────────────────────────────────

class ESM2Scorer:
    """
    Compute the mean masked pseudo-log-likelihood for a protein sequence
    using the ESM2-650M model (Rives et al. 2021; Lin et al. 2023).

    Higher PLL = the sequence is more "natural" according to the ESM2 model.
    We use this as a proxy for foldability / fitness.

    Reference: https://github.com/facebookresearch/esm
    """

    def __init__(self, device_str: str = "auto") -> None:
        import torch
        import esm  # fair-esm

        if device_str == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device_str)

        print(f"[step3] Loading ESM2-650M on {self._device} …")
        self._model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self._model = self._model.eval().to(self._device)
        self._batch_converter = self._alphabet.get_batch_converter()
        print("[step3] ESM2-650M ready.")

    def pll_mean(self, sequence: str) -> float:
        """Masked pseudo-log-likelihood per position (mean over all positions)."""
        import torch

        _, _, tokens = self._batch_converter([("protein", sequence)])
        tokens = tokens.to(self._device)

        mask_idx = self._alphabet.mask_idx
        eos_idx  = self._alphabet.eos_idx
        pad_idx  = self._alphabet.padding_idx

        log_probs: list[float] = []
        with torch.no_grad():
            for i in range(1, tokens.shape[1]):
                t_i = tokens[0, i].item()
                if t_i in (pad_idx, eos_idx):
                    break
                masked = tokens.clone()
                masked[0, i] = mask_idx
                out = self._model(masked, repr_layers=[], return_contacts=False)
                lp = torch.log_softmax(out["logits"][0, i], dim=-1)[tokens[0, i]]
                log_probs.append(lp.item())

        return float(np.mean(log_probs)) if log_probs else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3: ESM2 ranking and top-construct selection for eGFP"
    )
    ap.add_argument(
        "--mpnn_csv",
        type=Path,
        default=DEFAULT_CSV,
        help="CSV produced by Step 2 (mpnn_sequences.csv).",
    )
    ap.add_argument("--out_dir",       type=Path, default=OUT_DIR)
    ap.add_argument("--min_recovery",  type=float, default=MIN_RECOVERY,
                    help="Minimum overall sequence recovery (default 0.80).")
    ap.add_argument("--max_mutations", type=int,   default=MAX_MUTATIONS,
                    help="Maximum mutations vs. wild-type (default 25).")
    ap.add_argument("--n_export",      type=int,   default=N_EXPORT,
                    help="Number of top constructs to export (default 12).")
    ap.add_argument("--device",        type=str,   default="auto",
                    help="PyTorch device: auto | cpu | cuda | cuda:0.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load MPNN sequences ────────────────────────────────────────────────
    if not args.mpnn_csv.is_file():
        raise FileNotFoundError(
            f"Step 2 CSV not found: {args.mpnn_csv}\n"
            "Run step2_proteinmpnn_inverse_folding.py first."
        )
    df = pd.read_csv(args.mpnn_csv)
    print(f"[step3] Loaded {len(df)} sequences from {args.mpnn_csv}")

    # ── 2. Score wild-type as baseline ────────────────────────────────────────
    scorer = ESM2Scorer(device_str=args.device)

    print("[step3] Scoring wild-type EGFP …")
    wt_pll = scorer.pll_mean(EGFP_SEQUENCE)
    print(f"[step3] Wild-type PLL = {wt_pll:.4f} nat/position")

    # ── 3. Score all MPNN sequences ───────────────────────────────────────────
    plls: list[float] = []
    for i, seq in enumerate(df["sequence"], start=1):
        pll = scorer.pll_mean(str(seq))
        plls.append(pll)
        print(
            f"  [{i:3d}/{len(df)}] PLL={pll:.4f}  "
            f"n_mut={df.at[i-1,'n_mutations']:3d}  "
            f"rec={df.at[i-1,'sequence_recovery']:.1%}"
        )

    df["esm2_650M_pll_mean"] = plls
    df["wt_pll_baseline"]    = wt_pll
    df["pll_vs_wt"]          = df["esm2_650M_pll_mean"] - wt_pll

    # ── 4. Apply filters ──────────────────────────────────────────────────────
    df["passes_filter"] = (
        (df["sequence_recovery"] >= args.min_recovery)
        & (df["n_mutations"]     <= args.max_mutations)
        & (df["esm2_650M_pll_mean"] >= wt_pll)   # at least as natural as WT
    )
    n_pass = df["passes_filter"].sum()
    print(
        f"\n[step3] Filter: recovery≥{args.min_recovery:.0%}, "
        f"n_mut≤{args.max_mutations}, PLL≥WT → "
        f"{n_pass}/{len(df)} sequences pass."
    )

    # ── 5. Rank by ESM2 PLL (descending) ─────────────────────────────────────
    df = df.sort_values("esm2_650M_pll_mean", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    # ── 6. Save ranked CSV ────────────────────────────────────────────────────
    ranked_csv = args.out_dir / "ranked_sequences.csv"
    df.to_csv(ranked_csv, index=False)
    print(f"[step3] Ranked CSV → {ranked_csv}")

    # ── 7. Export top constructs (FASTA) ──────────────────────────────────────
    top_df = df[df["passes_filter"]].head(args.n_export)
    top_fasta = args.out_dir / "top_constructs.fasta"

    with top_fasta.open("w") as f:
        # Wild-type for reference
        f.write(
            f">wildtype_EGFP_4EUL|n_mut=0|recovery=1.0"
            f"|pll={wt_pll:.4f}\n{EGFP_SEQUENCE}\n"
        )
        for _, row in top_df.iterrows():
            f.write(
                f">egfp_watermark_{int(row['rank']):02d}"
                f"|n_mut={int(row['n_mutations'])}"
                f"|recovery={row['sequence_recovery']:.4f}"
                f"|pll={row['esm2_650M_pll_mean']:.4f}"
                f"|pll_vs_wt={row['pll_vs_wt']:+.4f}\n"
                f"{row['sequence']}\n"
            )
    print(f"[step3] Top-{args.n_export} FASTA → {top_fasta}  ({len(top_df)} constructs)")

    # ── 8. Summary statistics ─────────────────────────────────────────────────
    passing = df[df["passes_filter"]]
    print(
        f"\n=== eGFP watermark design — Step 3 summary ===\n"
        f"  Total MPNN sequences scored : {len(df)}\n"
        f"  Passing all filters         : {n_pass}\n"
        f"  Wild-type PLL (baseline)    : {wt_pll:.4f} nat/pos\n"
    )
    if len(passing):
        print(
            f"  Passing sequences:\n"
            f"    Mean PLL                : {passing['esm2_650M_pll_mean'].mean():.4f}\n"
            f"    Mean recovery           : {passing['sequence_recovery'].mean():.1%}\n"
            f"    Mean mutations          : {passing['n_mutations'].mean():.1f}\n"
        )
    print(f"  Exported for synthesis      : {len(top_df)} constructs → {top_fasta}\n")


if __name__ == "__main__":
    main()
