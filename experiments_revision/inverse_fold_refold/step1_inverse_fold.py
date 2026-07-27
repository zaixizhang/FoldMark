#!/usr/bin/env python3
"""
Step 1 — Inverse-fold the watermarked structures
=================================================

First half of the "launder through sequence space" round trip: take a
FoldMark-watermarked backbone and run ProteinMPNN over it to recover sequences.

The watermark lives in the **backbone geometry**, and a sequence carries no
explicit watermark bits.  Whether the mark survives this step therefore depends
entirely on whether the watermarked geometry biases which residues ProteinMPNN
chooses — and then, in Step 2, on whether a clean folding model reproduces that
geometry from the sequence alone.  This is the strongest form of the removal
attack: the attacker never touches the watermark, they simply re-derive the
structure through a channel that has no knowledge of it.

The whole chain is run at defaults; nothing here is tuned to preserve the mark.

Input
-----
CSV of watermarked structures:

    structure_id,pdb_path,message
    s0001,/data/wm/s0001.pdb,10110011010110100110100110101101

`message` is the 32-bit string embedded when the structure was produced; Step 3
needs it to count matching bits.

Output
------
outputs/step1/
    <structure_id>/seqs/<structure_id>.fa    raw ProteinMPNN output
    inverse_folded_sequences.csv             one row per (structure, MPNN sample)

Run:
    python experiments_revision/inverse_fold_refold/step1_inverse_fold.py \\
        --manifest watermarked_structures.csv --mpnn_dir ./ProteinMPNN
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step1"

MPNN_MODEL = "v_48_020"
MPNN_TEMP = 0.1          # low temperature: highest-fidelity inverse folding
N_SEQS = 8
MPNN_SEED = 42


def run_proteinmpnn(mpnn_dir: Path, pdb_path: Path, out_dir: Path,
                    num_seq: int, temperature: float, seed: int,
                    chain: str = "A") -> list[tuple[str, str]]:
    """Run ProteinMPNN over the whole chain (no fixed positions). Returns (header, seq)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(mpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", chain,
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", str(temperature),
        "--seed", str(seed),
        "--model_name", MPNN_MODEL,
        "--batch_size", "1",
        "--suppress_print", "1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mpnn_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(mpnn_dir), env=env, check=True)

    fa = out_dir / "seqs" / f"{pdb_path.stem}.fa"
    if not fa.is_file():
        raise FileNotFoundError(f"ProteinMPNN produced no FASTA at {fa}")

    entries, lines = [], fa.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">") and "sample=" in line and i + 1 < len(lines):
            entries.append((line[1:], lines[i + 1].strip()))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 1: inverse-fold watermarked structures with ProteinMPNN"
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV: structure_id,pdb_path,message")
    ap.add_argument("--mpnn_dir", type=Path,
                    default=Path(os.environ.get("MPNN_DIR", "./ProteinMPNN")))
    ap.add_argument("--out_dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--num_seq",  type=int, default=N_SEQS)
    ap.add_argument("--temperature", type=float, default=MPNN_TEMP)
    ap.add_argument("--seed",     type=int, default=MPNN_SEED)
    ap.add_argument("--chain",    type=str, default="A")
    ap.add_argument("--limit",    type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    for col in ("structure_id", "pdb_path", "message"):
        if col not in df.columns:
            sys.exit(f"[error] manifest is missing the {col!r} column")
    if args.limit:
        df = df.head(args.limit)

    mpnn_dir = args.mpnn_dir.resolve()
    if not (mpnn_dir / "protein_mpnn_run.py").is_file():
        sys.exit(f"[error] ProteinMPNN not found at {mpnn_dir}\n"
                 "        git clone https://github.com/dauparas/ProteinMPNN")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step1] {len(df)} watermarked structures x {args.num_seq} MPNN sequences "
          f"(T={args.temperature})")

    rows: list[dict] = []
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        sid = str(rec.structure_id)
        pdb = Path(str(rec.pdb_path))
        if not pdb.is_file():
            print(f"  [{i}/{len(df)}] {sid}: missing {pdb} — skipped")
            continue

        try:
            samples = run_proteinmpnn(
                mpnn_dir, pdb, args.out_dir / sid,
                args.num_seq, args.temperature, args.seed, args.chain,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(df)}] {sid}: FAILED ({type(exc).__name__}: {exc})")
            continue

        for k, (hdr, seq) in enumerate(samples):
            rows.append({
                "structure_id": sid,
                "mpnn_sample": k,
                "sequence": seq,
                "length": len(seq),
                "mpnn_header": hdr,
                "source_pdb": str(pdb),
                "message": str(rec.message),
            })
        print(f"  [{i}/{len(df)}] {sid}: {len(samples)} sequences")

    if not rows:
        sys.exit("[error] no sequences produced")

    out_csv = args.out_dir / "inverse_folded_sequences.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure_id","mpnn_sample","sequence","length",
            "mpnn_header","source_pdb","message"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[step1] {len(rows)} sequences from "
          f"{len(set(r['structure_id'] for r in rows))} structures\n"
          f"  → {out_csv}\n"
          f"\n  Pass to Step 2:  --sequences {out_csv}\n")


if __name__ == "__main__":
    main()
