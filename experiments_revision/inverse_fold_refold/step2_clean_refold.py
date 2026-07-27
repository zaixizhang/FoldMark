#!/usr/bin/env python3
"""
Step 2 — Re-fold the inverse-folded sequences with CLEAN folding models
=======================================================================

Second half of the round trip.  The sequences from Step 1 are folded again by
models that know nothing about FoldMark:

    protenix   Protenix with `watermark=False` — the same architecture as the
               watermarked predictor, but the watermarking path disabled
    esmfold    ESMFold (facebook/esmfold_v1) — a completely different model
    af3        AlphaFold3, via an external prediction command

"Clean" is the whole point: if a watermark is still recoverable from a structure
produced by a model that never saw the watermarking module, the mark must have
survived in the *sequence* chosen by inverse folding, and then been re-expressed
in the geometry.  A drop to chance here would mean the round trip removes it.

Every re-folded structure is also TM-aligned back to the original watermarked
backbone. That separates the two explanations for a lost watermark:

    high TM, low bit accuracy  → the mark itself was destroyed
    low  TM                    → the structure was not reproduced at all, so
                                 watermark loss says nothing about robustness

Output
------
outputs/step2/
    <folder>/<structure_id>_s<k>.pdb    re-folded structures
    refolded_manifest.csv               one row per (structure, sample, folder)

Run (protenix/af3 need the FoldMark-Protenix root on the path):
    python experiments_revision/inverse_fold_refold/step2_clean_refold.py \\
        --sequences experiments_revision/inverse_fold_refold/outputs/step1/inverse_folded_sequences.csv \\
        --folders esmfold,protenix
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step2"
FOLDERS = ["protenix", "esmfold", "af3"]

FOLDER_LABELS = {
    "original": "No re-folding",
    "protenix": "Protenix (clean)",
    "esmfold": "ESMFold",
    "af3": "AlphaFold3",
}


# ── structure helpers ─────────────────────────────────────────────────────────

def extract_ca(pdb_path: Path, chain_id: str = "A") -> tuple[str, np.ndarray]:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    model = structure[0]
    chain = model[chain_id] if chain_id in model else next(iter(model))
    seq, coords = [], []
    for residue in chain:
        if "CA" not in residue:
            continue
        seq.append(seq1(residue.get_resname()))
        coords.append(residue["CA"].get_coord())
    return "".join(seq), np.asarray(coords, dtype=float)


def tm_to_reference(ref_pdb: Path, mob_pdb: Path) -> float:
    from tmtools import tm_align

    ref_seq, ref_xyz = extract_ca(ref_pdb)
    mob_seq, mob_xyz = extract_ca(mob_pdb)
    if len(ref_xyz) == 0 or len(mob_xyz) == 0:
        return float("nan")
    res = tm_align(ref_xyz, mob_xyz, ref_seq, mob_seq)
    return float(max(res.tm_norm_chain1, res.tm_norm_chain2))


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO

    io = PDBIO()
    io.set_structure(MMCIFParser(QUIET=True).get_structure("s", str(cif_path)))
    io.save(str(pdb_path))
    return pdb_path


# ── folders ───────────────────────────────────────────────────────────────────

_ESMFOLD: dict = {}


def fold_esmfold(sequence: str, out_pdb: Path, device: str = "auto") -> Path:
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    if "model" not in _ESMFOLD:
        dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        print(f"[step2] Loading ESMFold on {dev} …")
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        mdl = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().to(dev)
        _ESMFOLD.update(model=mdl, tokenizer=tok, device=dev)

    mdl, tok, dev = _ESMFOLD["model"], _ESMFOLD["tokenizer"], _ESMFOLD["device"]
    inputs = tok([sequence], return_tensors="pt", add_special_tokens=False).to(dev)
    with torch.no_grad():
        output = mdl(**inputs)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(mdl.output_to_pdb(output)[0])
    return out_pdb


def fold_protenix_clean(sequence: str, name: str, work_dir: Path,
                        checkpoint: str, seed: int = 101) -> Path:
    """
    Protenix with the watermarking path disabled (`configs.watermark = False`).
    """
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import (
        InferenceRunner,
        download_infercence_cache,
        infer_predict,
    )
    from runner.msa_search import update_infer_json

    work_dir.mkdir(parents=True, exist_ok=True)
    json_path = work_dir / f"{name}.json"
    json_path.write_text(json.dumps(
        [{"sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
          "name": name}], indent=2))

    arg_str = (f"--seeds {seed} --dump_dir {work_dir} "
               f"--input_json_path {json_path} --model.N_cycle 10 "
               f"--sample_diffusion.N_sample 1 --sample_diffusion.N_step 200")
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str,
                            fill_required_with_null=True)
    configs.load_checkpoint_path = checkpoint
    download_infercence_cache()
    configs.input_json_path = update_infer_json(str(json_path), str(work_dir.parent), True)
    configs.watermark = False          # clean predictor
    configs.saved_path = str(work_dir)

    runner = InferenceRunner(configs)
    infer_predict(runner, configs)

    cifs = sorted((work_dir / "predictions").glob("**/*.cif"))
    if not cifs:
        raise FileNotFoundError(f"no CIF produced under {work_dir}")
    return cifs[0]


def fold_af3(sequence: str, name: str, work_dir: Path, command: str) -> Path:
    """
    AlphaFold3 via an external command.  `command` is a shell template with
    {sequence}, {name} and {out_dir} placeholders, e.g.

        --af3_command "python run_alphafold.py --sequence {sequence} \\
                       --name {name} --output_dir {out_dir}"

    AF3 has its own licensing and installation path, so it is intentionally not
    wrapped here beyond shelling out.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = command.format(sequence=sequence, name=name, out_dir=str(work_dir))
    subprocess.run(cmd, shell=True, check=True)

    produced = sorted(list(work_dir.glob("**/*.cif")) + list(work_dir.glob("**/*.pdb")))
    if not produced:
        raise FileNotFoundError(
            f"AF3 command produced no structure under {work_dir}; "
            "check the --af3_command template")
    return produced[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: re-fold inverse-folded sequences with clean models"
    )
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--out_dir",   type=Path, default=OUT_DIR)
    ap.add_argument("--folders",   type=str, default="protenix,esmfold")
    ap.add_argument("--checkpoint", type=str, default="./checkpoint.pt")
    ap.add_argument("--af3_command", type=str, default=None,
                    help="shell template with {sequence} {name} {out_dir}")
    ap.add_argument("--seed",  type=int, default=101)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    for f in folders:
        if f not in FOLDERS:
            sys.exit(f"[error] unknown folder {f!r}; choose from {FOLDERS}")
    if "af3" in folders and not args.af3_command:
        sys.exit("[error] --folders includes af3 but --af3_command was not given")

    df = pd.read_csv(args.sequences)
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step2] {len(df)} sequences x {len(folders)} folders: {folders}")

    rows: list[dict] = []
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        sid = str(rec.structure_id)
        k   = int(rec.mpnn_sample)
        seq = str(rec.sequence)
        tag = f"{sid}_s{k}"
        src = Path(str(rec.source_pdb))

        for folder in folders:
            dest = args.out_dir / folder / f"{tag}.pdb"
            ok, err = True, ""
            try:
                if folder == "esmfold":
                    fold_esmfold(seq, dest)
                elif folder == "protenix":
                    cif = fold_protenix_clean(seq, tag, args.out_dir / folder / tag,
                                              args.checkpoint, seed=args.seed)
                    cif_to_pdb(cif, dest)
                else:
                    produced = fold_af3(seq, tag, args.out_dir / folder / tag,
                                        args.af3_command)
                    if produced.suffix == ".cif":
                        cif_to_pdb(produced, dest)
                    else:
                        dest.write_text(produced.read_text())
            except Exception as exc:  # noqa: BLE001
                ok, err = False, f"{type(exc).__name__}: {exc}"

            tm = float("nan")
            if ok and src.is_file():
                try:
                    tm = tm_to_reference(src, dest)
                except Exception as exc:  # noqa: BLE001
                    err = f"TM-align failed: {type(exc).__name__}: {exc}"

            rows.append({
                "structure_id": sid, "mpnn_sample": k, "folder": folder,
                "pdb_path": str(dest) if ok else "",
                "source_pdb": str(src), "message": str(rec.message),
                "tm_to_watermarked": tm, "ok": ok, "error": err,
            })

        if i % 10 == 0 or i == len(df):
            print(f"  [{i}/{len(df)}] {tag}")

    out_csv = args.out_dir / "refolded_manifest.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "structure_id","mpnn_sample","folder","pdb_path","source_pdb",
            "message","tm_to_watermarked","ok","error"])
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["ok"] for r in rows)
    print(f"\n[step2] {n_ok}/{len(rows)} re-folds succeeded → {out_csv}")
    done = pd.DataFrame([r for r in rows if r["ok"]])
    if len(done):
        print("\n[step2] Mean TM-score to the watermarked backbone, by folder:")
        print(done.groupby("folder")["tm_to_watermarked"]
                  .agg(["count", "mean", "std"]).to_string())
    print(f"\n  Pass to Step 3:  --manifest {out_csv}\n")


if __name__ == "__main__":
    main()
