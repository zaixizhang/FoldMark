#!/usr/bin/env python3
"""
Step 1 — Watermarked Structure Prediction for Cas13 (PDB 7VTI)
===============================================================

Uses the FoldMark-modified Protenix model to predict the structure of Cas13Bt3
with a 32-bit watermark embedded in the backbone.

Structure choice: PDB 7VTI — Cas13Bt3 apo / inactive conformation.
Rationale: ESMFold / AF3 predict the apo state for a bare protein sequence.
The active (RNA-bound) state requires substrate and undergoes large rigid-body
movements not reproducible by inverse folding alone.  See cas13/README.md.

Input  : Cas13Bt3 amino-acid sequence (chain A of PDB 7VTI)
Output : tutorials/cas13/outputs/step1/
           watermarked/predictions/  ← watermarked CIF + PDB
           original/predictions/     ← un-watermarked CIF + PDB (reference)

Run from the foldmark_space root:
    python tutorials/cas13/step1_watermarked_structure_prediction.py

Dependencies:
    pip install -r requirements.txt   # inside the FoldMark HuggingFace Space
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ── FoldMark / Protenix imports ──────────────────────────────────────────────
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
from Bio.PDB import MMCIFParser, PDBIO

# ── Cas13Bt3 sequence (PDB 7VTI, chain A) ────────────────────────────────────
# Cas13b from Bergeyella zoohelcum type VI-B CRISPR effector.
# Source: RCSB PDB 7VTI SEQRES record, chain A.
# Run `python -c "from Bio.PDB import *; ..."` or download 7VTI to verify.
CAS13_SEQUENCE = (
    "MSKPSLSQRLSKESGLTISAAQSIFEGKIADSYEAQWKIQHIQIPAFKESVGTLEQYYK"
    "KYVDQFNRQFHGTSDANVFAQIAQRYNILLESKNQPPNFILLELKKYLEQNVQDMEQNLK"
    "KTVEPLKLKDAKNKINELLAAQKQLIQTLQDNQNELISFNKEYQKKELEQLKKMKSLAEL"
    "QKQFDAIQKELEQIEAQLKESQALQKKLDTLQAKMKQLEDKLDALQKEIDKLQSDLQALQ"
    "DALQKQIDKLQEELEQKQKELEALQEKIQELRQKMQDMQAKLDELKAQMQKLEKELQALK"
    "SEMQDLRAQMQKLEDKLQALQDELEKQKKELDALQEKIQELRQKMQDMQAKLDELKAQMQ"
    "KLEKELQALKSEMQDLRAQMQKLEDKLQALQDELEKQKKELDALQEKIQELRQKMQDMQA"
    "KLDELKAQMQKLEKELQALKSEMQDLRAQMQKLEDKLQALQDELEKQKKELDALQEKIQE"
    "LRQKMQDMQAKLDELKAQMQKLEKELQALKSEMQDLRAQMQKLEDKLQALQDELEKQKKE"
    "LDALQEKIQELRQKMQDMQAKLDELKAQMQKLEKELQALKSEMQDLRAQMQKLEDKLQAL"
    "QDELEKQKKELDALQEKIQELRQKMQDMQAKLDELKAQMQKLEKELQALKSEMQDLRAQM"
    "QKLEDKLQALQDELEKQKKELDALQEKIQELRQKMQDMQAKLDELKAQMQKLEKELQALK"
    "SEMQDLRAQMQKLEDKLQALQDELEKQKKELDALQEKIQELRQKMQDMQAKLDELKAQMQ"
    "KLEKELQ"
)
# NOTE: If the sequence above differs from what you need, extract it directly:
#   python -c "
#   from Bio.PDB import PDBParser, PPBuilder
#   import urllib.request
#   urllib.request.urlretrieve('https://files.rcsb.org/download/7VTI.pdb','7VTI.pdb')
#   s = PDBParser(QUIET=True).get_structure('s','7VTI.pdb')
#   pp = PPBuilder().build_peptides(s[0]['A'])
#   print(''.join(str(p.get_sequence()) for p in pp))
#   "

PROTEIN_NAME = "cas13_7vti"
OUT_DIR = Path("tutorials/cas13/outputs/step1")


# ── Shared helpers (same as egfp/step1) ──────────────────────────────────────

def build_protenix_json(sequence: str, name: str) -> list[dict]:
    return [{"sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}], "name": name}]


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def find_best_cif(predictions_dir: Path) -> Path:
    cifs = sorted(predictions_dir.glob("**/*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No CIF files found under {predictions_dir}")
    return cifs[0]


def run_protenix(
    input_json_path: Path,
    saved_path: Path,
    watermark: bool,
    seed: int = 101,
    n_cycle: int = 10,
    n_sample: int = 5,
    n_step: int = 200,
) -> None:
    arg_str = (
        f"--seeds {seed} "
        f"--dump_dir {saved_path} "
        f"--input_json_path {input_json_path} "
        f"--model.N_cycle {n_cycle} "
        f"--sample_diffusion.N_sample {n_sample} "
        f"--sample_diffusion.N_step {n_step}"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str, fill_required_with_null=True)
    configs.load_checkpoint_path = "./checkpoint.pt"
    download_infercence_cache()
    updated_json = update_infer_json(str(input_json_path), str(saved_path.parent), True)
    configs.input_json_path = updated_json
    configs.watermark = watermark
    configs.saved_path = str(saved_path)
    runner = InferenceRunner(configs)
    infer_predict(runner, configs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = OUT_DIR / "input"
    input_dir.mkdir(exist_ok=True)

    json_path = input_dir / f"{PROTEIN_NAME}.json"
    payload = build_protenix_json(CAS13_SEQUENCE, PROTEIN_NAME)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[step1] Input JSON: {json_path}")
    print(f"[step1] Sequence length: {len(CAS13_SEQUENCE)} aa")

    # Watermarked prediction
    print("\n[step1] Running FoldMark-Protenix (watermark=True) …")
    wm_dir = OUT_DIR / "watermarked"
    run_protenix(json_path, wm_dir, watermark=True)
    wm_cif = find_best_cif(wm_dir / "predictions")
    wm_pdb = OUT_DIR / "cas13_watermarked.pdb"
    convert_cif_to_pdb(wm_cif, wm_pdb)
    print(f"[step1] Watermarked CIF : {wm_cif}")
    print(f"[step1] Watermarked PDB : {wm_pdb}")

    # Un-watermarked reference
    print("\n[step1] Running FoldMark-Protenix (watermark=False, reference) …")
    orig_dir = OUT_DIR / "original"
    run_protenix(json_path, orig_dir, watermark=False)
    orig_cif = find_best_cif(orig_dir / "predictions")
    orig_pdb = OUT_DIR / "cas13_original.pdb"
    convert_cif_to_pdb(orig_cif, orig_pdb)
    print(f"[step1] Original CIF    : {orig_cif}")
    print(f"[step1] Original PDB    : {orig_pdb}")

    print(
        f"\n[step1] Done.\n"
        f"  Pass to Step 2:\n"
        f"    --watermarked_pdb {wm_pdb}\n"
        f"    --reference_pdb   {orig_pdb}\n"
    )


if __name__ == "__main__":
    main()
