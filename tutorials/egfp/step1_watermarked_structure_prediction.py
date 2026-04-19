#!/usr/bin/env python3
"""
Step 1 — Watermarked Structure Prediction for eGFP (PDB 4EUL)
==============================================================

Uses the FoldMark-modified Protenix model to predict the structure of EGFP
with a 32-bit watermark embedded in the backbone.

Input  : EGFP amino-acid sequence (chain A of PDB 4EUL, 239 residues)
Output : tutorials/egfp/outputs/step1/
           watermarked/predictions/  ← watermarked CIF + PDB
           original/predictions/     ← un-watermarked CIF + PDB (reference)

Run from the foldmark_space root:
    python tutorials/egfp/step1_watermarked_structure_prediction.py

Dependencies (install once):
    pip install -r requirements.txt   # inside the FoldMark HuggingFace Space
    # checkpoint.pt is downloaded automatically on first run
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
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
    update_inference_configs,
)
from runner.msa_search import update_infer_json
from Bio.PDB import MMCIFParser, PDBIO

# ── eGFP sequence (PDB 4EUL, chain A, 239 residues) ─────────────────────────
# Enhanced GFP with S65T / H148D mutations.
# Source: RCSB PDB 4EUL SEQRES record, chain A.
EGFP_SEQUENCE = (
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWP"
    "TLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDT"
    "LVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQL"
    "ADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)

PROTEIN_NAME = "egfp_4eul"
OUT_DIR = Path("tutorials/egfp/outputs/step1")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_protenix_json(sequence: str, name: str) -> list[dict]:
    """Create a minimal Protenix input JSON for a single protein chain."""
    return [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": name,
        }
    ]


def convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> None:
    """Convert a mmCIF file to PDB format (using Biopython)."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))


def find_best_cif(predictions_dir: Path) -> Path:
    """Return the CIF file from the predictions directory (sorted, first entry)."""
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
    """
    Run Protenix (FoldMark-modified) structure prediction.

    Parameters
    ----------
    input_json_path : Path
        JSON file describing the protein complex (Protenix format).
    saved_path : Path
        Root directory where predictions will be saved.
    watermark : bool
        If True, embed the FoldMark 32-bit watermark in the backbone.
    """
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

    # Download checkpoint + CCD cache on first run
    download_infercence_cache()

    # MSA search / cache update
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

    # Write Protenix JSON
    json_path = input_dir / f"{PROTEIN_NAME}.json"
    payload = build_protenix_json(EGFP_SEQUENCE, PROTEIN_NAME)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[step1] Input JSON: {json_path}")
    print(f"[step1] Sequence length: {len(EGFP_SEQUENCE)} aa")

    # ── Watermarked prediction ────────────────────────────────────────────────
    print("\n[step1] Running FoldMark-Protenix (watermark=True) …")
    wm_dir = OUT_DIR / "watermarked"
    run_protenix(json_path, wm_dir, watermark=True)

    wm_cif = find_best_cif(wm_dir / "predictions")
    wm_pdb = OUT_DIR / "egfp_watermarked.pdb"
    convert_cif_to_pdb(wm_cif, wm_pdb)
    print(f"[step1] Watermarked CIF : {wm_cif}")
    print(f"[step1] Watermarked PDB : {wm_pdb}")

    # ── Un-watermarked reference (optional; used for TM-score comparison) ─────
    print("\n[step1] Running FoldMark-Protenix (watermark=False, reference) …")
    orig_dir = OUT_DIR / "original"
    run_protenix(json_path, orig_dir, watermark=False)

    orig_cif = find_best_cif(orig_dir / "predictions")
    orig_pdb = OUT_DIR / "egfp_original.pdb"
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
