#!/usr/bin/env python3
"""
Step 2 — Decode the watermark from every post-processed structure
==================================================================

Runs the FoldMark decoder on each structure produced by Step 1 and records how
many of the K embedded bits survive.

Decoding mirrors `runner.inference.infer_detect`:

    prediction['watermark'] → clamp(-10, 10) → sigmoid → threshold at 0.5

The number of bits K is taken from the decoder output rather than assumed, so a
checkpoint trained with a different payload width is handled correctly; FoldMark
itself uses K = 32 throughout (`watermark_emb: 32`).  Note that a TPR at
FPR = 1e-5 is only meaningful when K is large enough for the null tail to reach
it: at K = 32 the smallest attainable analytic FPR is 2^-32 ≈ 2.3e-10, with
plenty of headroom.  Step 3 warns if the requested FPR is below what K supports.

Output
------
outputs/step2/detection_scores.csv — one row per (structure, method):
    structure_id, dataset, method, n_bits, n_matching_bits, bit_accuracy

Run from the FoldMark-Protenix root, so that `configs/` and `runner/` import:
    python experiments_revision/robustness_postprocessing/step2_detect_watermark.py \\
        --manifest experiments_revision/robustness_postprocessing/outputs/step1/postprocessed_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent / "outputs" / "step1" / "postprocessed_manifest.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step2"


def load_runner(checkpoint: str = "./checkpoint.pt"):
    """Build a FoldMark InferenceRunner once and reuse it across structures."""
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_path = checkpoint
    return InferenceRunner(configs), configs


def decode_bits(runner, configs, pdb_path: Path) -> np.ndarray:
    """
    Decode the watermark bit string from one structure.

    Returns a 1-D array of 0/1 of length K (the model's code width).
    """
    import torch
    from runner.inference import get_inference_dataloader

    configs.input_json_path = str(pdb_path)
    dataloader = get_inference_dataloader(configs=configs)
    for batch in dataloader:
        prediction, _label = runner.detect(batch)
        code = torch.clamp(prediction["watermark"], min=-10, max=10)
        return (torch.sigmoid(code) > 0.5).long().cpu().numpy().ravel()
    raise RuntimeError(f"decoder produced no output for {pdb_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: decode the FoldMark watermark from post-processed structures"
    )
    ap.add_argument("--manifest",   type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--checkpoint", type=str,  default="./checkpoint.pt")
    ap.add_argument("--limit",      type=int,  default=None)
    args = ap.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"[error] Step 1 manifest not found: {args.manifest}\n"
                 "        Run step1_postprocess_structures.py first.")

    df = pd.read_csv(args.manifest)
    df = df[df["ok"].astype(str).str.lower() == "true"]
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step2] Decoding {len(df)} post-processed structures")

    runner, configs = load_runner(args.checkpoint)

    rows: list[dict] = []
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        pdb_path = Path(str(rec.pdb_path))
        true_bits = np.array([int(b) for b in str(rec.message).strip()], dtype=int)

        try:
            pred_bits = decode_bits(runner, configs, pdb_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(df)}] {rec.structure_id} {rec.method}: "
                  f"decode failed ({type(exc).__name__}: {exc})")
            continue

        k = min(pred_bits.size, true_bits.size)
        if k == 0:
            continue
        n_match = int((pred_bits[:k] == true_bits[:k]).sum())
        rows.append({
            "structure_id": rec.structure_id,
            "dataset": rec.dataset,
            "method": rec.method,
            "iterations": rec.iterations,
            "n_bits": k,
            "n_matching_bits": n_match,
            "bit_accuracy": n_match / k,
        })

        if i % 20 == 0 or i == len(df):
            print(f"  [{i}/{len(df)}] {rec.structure_id:<20s} {rec.method:<18s} "
                  f"{n_match}/{k} bits")

    if not rows:
        sys.exit("[error] no structures decoded successfully")

    out = pd.DataFrame(rows)
    out_csv = args.out_dir / "detection_scores.csv"
    out.to_csv(out_csv, index=False)

    print(f"\n[step2] Mean bit accuracy by dataset and method:")
    print(out.groupby(["dataset", "method"])["bit_accuracy"]
             .agg(["count", "mean", "std"]).to_string())
    print(f"\n[step2] Scores → {out_csv}\n"
          f"\n  Pass to Step 3:  --scores_csv {out_csv}\n")


if __name__ == "__main__":
    main()
