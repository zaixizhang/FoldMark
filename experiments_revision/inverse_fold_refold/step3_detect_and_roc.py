#!/usr/bin/env python3
"""
Step 3 — Detect the 32-bit watermark after the round trip, and build the ROC
=============================================================================

Runs the FoldMark decoder on every re-folded structure from Step 2 and asks
whether the **32-bit** message is still recoverable after inverse folding plus
clean re-folding.  Decoding is unchanged from normal detection — the same
decoder, the same 32 bits, no re-training and no knowledge of which folder
produced the structure:

    prediction['watermark'] → clamp(-10, 10) → sigmoid → threshold at 0.5

Conditions compared (from Step 2, plus the untouched baseline):

    original   the watermarked structure itself, no round trip
    protenix   inverse fold → Protenix with watermarking disabled
    esmfold    inverse fold → ESMFold
    af3        inverse fold → AlphaFold3

Detection statistic and null
----------------------------
The statistic is k, the number of the 32 bits matching the embedded message.
Under H0 the bits are fair coins, so

    FPR(tau) = P[X >= tau],  X ~ Binomial(32, 0.5)

is analytic, which is what allows an FPR of 1e-5 to be quoted without 10^5
negative structures. **TPR is empirical** — the fraction of round-tripped
structures whose k reaches the threshold — so every plotted point is measured.

Per structure the best MPNN sample is used by default (`--aggregate best`),
matching the threat model in which an attacker keeps whichever inverse-folded
sequence re-folds most faithfully; `--aggregate mean` and `all` are also
available.

TM-score is carried through from Step 2 so that a low bit accuracy can be
attributed correctly: high TM with low bit accuracy means the mark was removed,
whereas low TM means the structure was never reproduced and the round trip says
nothing about watermark robustness.

Output
------
outputs/step3/
    detection_scores.csv    per structure/sample/folder: bits matched, TM
    roc_points.csv          (folder, tau, FPR, TPR)
    tpr_at_fpr.csv          TPR at the target FPR, with n per folder
    inverse_refold_roc.png / .pdf

Run from the FoldMark-Protenix root:
    python experiments_revision/inverse_fold_refold/step3_detect_and_roc.py \\
        --manifest experiments_revision/inverse_fold_refold/outputs/step2/refolded_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_clean_refold import FOLDER_LABELS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step3"

N_BITS = 32
CONDITION_ORDER = ["original", "protenix", "esmfold", "af3"]
COLORS  = ["silver", "darkorange", "darkcyan", "violet"]
MARKERS = ["o", "*", "p", "h"]


def load_runner(checkpoint: str):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_path = checkpoint
    return InferenceRunner(configs), configs


def decode_bits(runner, configs, pdb_path: Path, n_bits: int = N_BITS) -> np.ndarray:
    import torch
    from runner.inference import get_inference_dataloader

    configs.input_json_path = str(pdb_path)
    for batch in get_inference_dataloader(configs=configs):
        prediction, _label = runner.detect(batch)
        code = torch.clamp(prediction["watermark"], min=-10, max=10)
        return (torch.sigmoid(code) > 0.5).long().cpu().numpy().ravel()[:n_bits]
    raise RuntimeError(f"decoder produced no output for {pdb_path}")


def analytic_fpr(K: int):
    from scipy.stats import binom

    tau = np.arange(0, K + 1)
    return tau, 1.0 - binom.cdf(tau - 1, K, 0.5)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 3: detect the 32-bit watermark after inverse fold + clean re-fold"
    )
    ap.add_argument("--manifest",   type=Path, required=True)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--checkpoint", type=str,  default="./checkpoint.pt")
    ap.add_argument("--target_fpr", type=float, default=1e-5)
    ap.add_argument("--aggregate",  choices=["best", "mean", "all"], default="best",
                    help="how to combine the MPNN samples of one structure")
    ap.add_argument("--include_original", action="store_true", default=True,
                    help="also decode the untouched watermarked structures")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"[error] Step 2 manifest not found: {args.manifest}\n"
                 "        Run step2_clean_refold.py first.")

    df = pd.read_csv(args.manifest)
    df = df[df["ok"].astype(str).str.lower() == "true"]
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runner, configs = load_runner(args.checkpoint)

    jobs: list[dict] = []
    for rec in df.itertuples(index=False):
        jobs.append({"structure_id": rec.structure_id, "mpnn_sample": rec.mpnn_sample,
                     "folder": rec.folder, "pdb_path": rec.pdb_path,
                     "message": rec.message, "tm": rec.tm_to_watermarked})

    if args.include_original:
        seen = set()
        for rec in df.itertuples(index=False):
            src = str(rec.source_pdb)
            if src and src not in seen and Path(src).is_file():
                seen.add(src)
                jobs.append({"structure_id": rec.structure_id, "mpnn_sample": -1,
                             "folder": "original", "pdb_path": src,
                             "message": rec.message, "tm": 1.0})

    print(f"[step3] Decoding {len(jobs)} structures ({N_BITS}-bit watermark)")

    rows: list[dict] = []
    for i, job in enumerate(jobs, start=1):
        true_bits = np.array([int(c) for c in str(job["message"]).strip()], dtype=int)
        try:
            pred = decode_bits(runner, configs, Path(job["pdb_path"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(jobs)}] {job['structure_id']} {job['folder']}: "
                  f"decode failed ({type(exc).__name__}: {exc})")
            continue
        k = min(pred.size, true_bits.size)
        n_match = int((pred[:k] == true_bits[:k]).sum())
        rows.append({**{key: job[key] for key in
                        ("structure_id", "mpnn_sample", "folder", "tm")},
                     "n_bits": k, "n_matching_bits": n_match,
                     "bit_accuracy": n_match / k,
                     "recovered_bits": "".join(str(int(b)) for b in pred[:k])})
        if i % 20 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] {job['structure_id']:<14s} "
                  f"{job['folder']:<10s} {n_match}/{k} bits")

    if not rows:
        sys.exit("[error] nothing decoded successfully")

    scores = pd.DataFrame(rows)
    scores.to_csv(args.out_dir / "detection_scores.csv", index=False)

    # ── aggregate the MPNN samples of each structure ─────────────────────────
    if args.aggregate == "best":
        agg = (scores.sort_values("n_matching_bits", ascending=False)
                     .groupby(["structure_id", "folder"], as_index=False).first())
    elif args.aggregate == "mean":
        agg = (scores.groupby(["structure_id", "folder"], as_index=False)
                     .agg(n_matching_bits=("n_matching_bits", "mean"),
                          n_bits=("n_bits", "first"),
                          bit_accuracy=("bit_accuracy", "mean"),
                          tm=("tm", "mean")))
        agg["n_matching_bits"] = agg["n_matching_bits"].round().astype(int)
    else:
        agg = scores

    K = int(agg["n_bits"].mode().iat[0])
    tau, fpr = analytic_fpr(K)
    min_fpr = float(fpr[-1])
    if args.target_fpr < min_fpr:
        print(f"[step3] WARNING: target FPR {args.target_fpr:g} is below 2^-{K} "
              f"= {min_fpr:.3g}; reporting at {min_fpr:.3g}")

    conditions = [c for c in CONDITION_ORDER if c in set(agg["folder"])]
    roc_rows, op_rows = [], []
    for cond in conditions:
        sub = agg[agg["folder"] == cond]
        kv = sub["n_matching_bits"].to_numpy()
        tpr = np.array([(kv >= t).mean() for t in tau])
        for t, f, p in zip(tau, fpr, tpr):
            roc_rows.append({"folder": cond, "tau": int(t),
                             "fpr": float(f), "tpr": float(p)})
        usable = np.where(fpr <= max(args.target_fpr, min_fpr))[0]
        idx = int(usable[0]) if usable.size else len(tau) - 1
        op_rows.append({
            "folder": cond, "label": FOLDER_LABELS.get(cond, cond),
            "n_structures": int(len(sub)), "n_bits": K,
            "threshold_tau": int(tau[idx]), "achieved_fpr": float(fpr[idx]),
            "tpr": float(tpr[idx]),
            "mean_bit_accuracy": float(sub["bit_accuracy"].mean()),
            "sd_bit_accuracy": float(sub["bit_accuracy"].std(ddof=1))
                               if len(sub) > 1 else float("nan"),
            "mean_tm_to_watermarked": float(sub["tm"].mean()),
        })

    pd.DataFrame(roc_rows).to_csv(args.out_dir / "roc_points.csv", index=False)
    ops = pd.DataFrame(op_rows)
    ops.to_csv(args.out_dir / "tpr_at_fpr.csv", index=False)

    print(f"\n[step3] K = {K} bits, aggregate = {args.aggregate}, "
          f"TPR at FPR <= {args.target_fpr:g}:")
    print(ops[["label","n_structures","threshold_tau","achieved_fpr","tpr",
               "mean_bit_accuracy","mean_tm_to_watermarked"]].to_string(index=False))

    # ── figure ───────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]

    roc = pd.DataFrame(roc_rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for cond, color, marker in zip(conditions, COLORS, MARKERS):
        sub = roc[(roc["folder"] == cond) & (roc["fpr"] > 0)]
        if sub.empty:
            continue
        ax.plot(sub["fpr"], sub["tpr"], label=FOLDER_LABELS.get(cond, cond),
                color=color, marker=marker, markersize=5, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=14)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=14)
    ax.set_ylim(-0.03, 1.05)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12, width=1.5)
    ax.yaxis.set_tick_params(labelsize=12, width=1.5)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    png = args.out_dir / "inverse_refold_roc.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / "inverse_refold_roc.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\n[step3] Figure → {png}")
    print("[step3] TPR is empirical over the round-tripped structures; FPR is the\n"
          "        analytic Binomial(32, 0.5) tail. Read mean_tm_to_watermarked\n"
          "        alongside the TPR: a low TPR at low TM means the structure was\n"
          "        not reproduced, not that the watermark was removed.\n")


if __name__ == "__main__":
    main()
