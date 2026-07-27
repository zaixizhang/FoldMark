#!/usr/bin/env python3
"""
Step 1 — Watermarked Structure Prediction for TEV Protease (PDB 1LVM)
======================================================================

Uses the FoldMark-modified Protenix model to predict the structure of the
Tobacco Etch Virus NIa protease (TEV) with a 32-bit watermark embedded in the
backbone.

Structure choice: PDB 1LVM — TEV NIa catalytic domain, chain A, residues 1–221.
The construct used here is **241 aa**: the 221-residue crystallographic core
plus a **20-residue engineered C-terminal extension (222–241)** that is absent
from the crystal.  Step 1 therefore predicts the *full-length* construct, which
is what makes the C-terminal extension available as a watermark carrier in
Step 2 (design group B) — the crystal structure alone cannot provide it.

Input  : TEV protease amino-acid sequence (241 aa construct)
Output : tutorials/tev/outputs/step1/
           watermarked/predictions/  ← watermarked CIF + PDB
           original/predictions/     ← un-watermarked CIF + PDB (reference)

Run from the foldmark_space root:
    python tutorials/tev/step1_watermarked_structure_prediction.py

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

# ── TEV protease sequence (1LVM chain A + engineered C-terminal extension) ───
# Positions 1–221  : TEV NIa catalytic domain, matches 1LVM chain A numbering
#                    (position 1 = the G of "GES…").
# Positions 222–241: engineered C-terminal extension, not present in the crystal.
#
# Known differences of this construct vs. the 1LVM crystal: Q74H.
TEV_SEQUENCE = (
    "GESLFKGPRDYNPISSTICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLLVQSL"
    "HGVFKVKNTTTLQHHLIDGRDMIIIRMPKDFPPFPQKLKFREPQREERICLVTTNFQTKS"
    "MSSMVSDTSCTFPSSDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFT"
    "SVPKNFMELLTNQEAQQWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQLMNRRRRR"
)
# NOTE: to re-extract the crystallographic core (1–221) directly:
#   python -c "
#   from Bio.PDB import PDBParser, PPBuilder
#   import urllib.request
#   urllib.request.urlretrieve('https://files.rcsb.org/download/1LVM.pdb','1LVM.pdb')
#   s = PDBParser(QUIET=True).get_structure('s','1LVM.pdb')
#   pp = PPBuilder().build_peptides(s[0]['A'])
#   print(''.join(str(p.get_sequence()) for p in pp))
#   "

PROTEIN_NAME = "tev_1lvm"
OUT_DIR = Path("tutorials/tev/outputs/step1")


# ── Shared helpers (same as egfp/step1, cas13/step1, dhfr/step1) ─────────────

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
    payload = build_protenix_json(TEV_SEQUENCE, PROTEIN_NAME)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[step1] Input JSON: {json_path}")
    print(f"[step1] Sequence length: {len(TEV_SEQUENCE)} aa "
          f"(1–221 crystallographic core + 222–241 engineered extension)")

    # Watermarked prediction
    print("\n[step1] Running FoldMark-Protenix (watermark=True) …")
    wm_dir = OUT_DIR / "watermarked"
    run_protenix(json_path, wm_dir, watermark=True)
    wm_cif = find_best_cif(wm_dir / "predictions")
    wm_pdb = OUT_DIR / "tev_watermarked.pdb"
    convert_cif_to_pdb(wm_cif, wm_pdb)
    print(f"[step1] Watermarked CIF : {wm_cif}")
    print(f"[step1] Watermarked PDB : {wm_pdb}")

    # Un-watermarked reference
    print("\n[step1] Running FoldMark-Protenix (watermark=False, reference) …")
    orig_dir = OUT_DIR / "original"
    run_protenix(json_path, orig_dir, watermark=False)
    orig_cif = find_best_cif(orig_dir / "predictions")
    orig_pdb = OUT_DIR / "tev_original.pdb"
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
