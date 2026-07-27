#!/usr/bin/env python3
"""
Step 1 — Apply routine post-processing to watermarked bound-state structures
=============================================================================

Runs each post-processing operator over a set of FoldMark-watermarked
structures, in **bound-state contexts**: protein–ligand complexes and
protein–antibody complexes (the same datasets used for Figure 4 of the main
paper).

Operators (all at tool defaults — see postprocess_ops.py):

  none                 baseline, unprocessed
  amber                Amber relaxation (OpenFold/AlphaFold relaxer)
  rosetta_fastrelax    Rosetta FastRelax
  rosetta_ngk          Next-generation KIC loop modeling

`--iterations N` applies the operator N times in sequence, feeding each output
back in, to cover the "single or iterative runs" case.

Input
-----
A manifest CSV describing the watermarked structures, with columns:

    structure_id,dataset,pdb_path,message
    7abc_A,ligand,/path/7abc_wm.pdb,1011001101011010...
    ...

`dataset` is `ligand` or `antibody`; `message` is the ground-truth bit string
that was embedded when the structure was produced (needed by Step 2 to count
matching bits).

Output
------
outputs/step1/
  <dataset>/<structure_id>/<method>.pdb    post-processed structures
  postprocessed_manifest.csv               one row per (structure, method)

Run:
    python experiments_revision/robustness_postprocessing/step1_postprocess_structures.py \\
        --manifest my_watermarked_structures.csv \\
        --methods none,amber,rosetta_fastrelax,rosetta_ngk
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_ops import METHODS, apply_method  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step1"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 1: apply Amber / Rosetta FastRelax / NGK to watermarked complexes"
    )
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV: structure_id,dataset,pdb_path,message")
    ap.add_argument("--out_dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--methods",  type=str, default=",".join(METHODS))
    ap.add_argument("--iterations", type=int, default=1,
                    help="apply each operator this many times in sequence")
    ap.add_argument("--params", type=str, nargs="*", default=None,
                    help="Rosetta ligand .params files")
    ap.add_argument("--amber_gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            sys.exit(f"[error] unknown method {m!r}; choose from {METHODS}")

    df = pd.read_csv(args.manifest)
    required = {"structure_id", "dataset", "pdb_path", "message"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[error] manifest is missing column(s): {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step1] {len(df)} structures x {len(methods)} methods "
          f"(iterations={args.iterations})")

    rows: list[dict] = []
    for _, rec in df.iterrows():
        sid     = str(rec["structure_id"])
        dataset = str(rec["dataset"])
        src     = Path(str(rec["pdb_path"]))
        if not src.is_file():
            print(f"  [skip] {sid}: missing {src}")
            continue

        dest_dir = args.out_dir / dataset / sid
        dest_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            t0 = time.time()
            cur_in = src
            ok, msg = True, ""
            out_pdb = dest_dir / f"{method}.pdb"

            for it in range(args.iterations):
                stage_out = (
                    out_pdb if it == args.iterations - 1
                    else dest_dir / f"{method}_iter{it+1}.pdb"
                )
                res = apply_method(
                    method, cur_in, stage_out,
                    params=args.params, use_gpu=args.amber_gpu,
                )
                ok, msg = res.ok, res.message
                if not ok:
                    break
                cur_in = stage_out

            dt = time.time() - t0
            rows.append({
                "structure_id": sid, "dataset": dataset, "method": method,
                "iterations": args.iterations,
                "pdb_path": str(out_pdb) if ok else "",
                "source_pdb": str(src), "message": str(rec["message"]),
                "ok": ok, "error": msg, "seconds": round(dt, 2),
            })
            status = "ok" if ok else f"FAILED ({msg[:60]})"
            print(f"  {sid:<20s} {method:<18s} {dt:7.1f}s  {status}")

    out_csv = args.out_dir / "postprocessed_manifest.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure_id","dataset","method","iterations","pdb_path",
            "source_pdb","message","ok","error","seconds"])
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["ok"] for r in rows)
    print(
        f"\n[step1] Done. {n_ok}/{len(rows)} post-processing runs succeeded.\n"
        f"  Manifest → {out_csv}\n"
        f"\n  Pass to Step 2:  --manifest {out_csv}\n"
    )
    if n_ok < len(rows):
        print("  Failures are recorded in the manifest with ok=False and the\n"
              "  exception text; Step 2 skips them and Step 3 reports the n used.\n")


if __name__ == "__main__":
    main()
