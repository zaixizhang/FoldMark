#!/usr/bin/env python3
"""
Step 2 — Fold both arms and score watermark recovery + structural quality
==========================================================================

For every matched pair produced by Step 1:

  chen arm
      The sequence is already watermarked at the sequence level.
      · Fold with the ORIGINAL structure predictor (Protenix / AF3,
        `watermark=False`).
      · Recover bits by scoring the token sequence under every candidate key
        (chen_watermark.ChenMultiBitWatermarker.decode).

  foldmark arm
      The sequence carries no watermark.
      · Fold with FoldMark-Protenix (`watermark=True`), which embeds the 32-bit
        message into the backbone.
      · Recover bits from the STRUCTURE via the FoldMark decoder
        (`InferenceRunner.detect` → sigmoid(prediction['watermark']) > 0.5).

Both arms are then scored for structural quality with **scTM**
(self-consistency TM-score): the predicted backbone is inverse-folded with
ProteinMPNN (8 sequences), each is re-folded with ESMFold, and scTM is the max
TM-score against the predicted backbone.

Output : outputs/step2/
           per_sequence_scores.csv   ← one row per (seq_id, arm)
           paired_scores.csv         ← one row per pair, both arms side by side

Run from the FoldMark-Protenix root (the HuggingFace Space checkout), so that
`configs/` and `runner/` are importable:
    python experiments_revision/chen_et_al_comparison/step2_fold_and_score.py \\
        --manifest experiments_revision/chen_et_al_comparison/outputs/step1/manifest.csv \\
        --mpnn_dir ./ProteinMPNN
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chen_watermark import (  # noqa: E402
    DEFAULT_N_CHUNKS,
    ChenMultiBitConfig,
    ChenMultiBitWatermarker,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "outputs" / "step1" / "manifest.csv"
OUT_DIR          = Path(__file__).resolve().parent / "outputs" / "step2"

N_MPNN_SEQS   = 8      # sequences per backbone for the scTM self-consistency loop
MPNN_TEMP     = 0.1
MPNN_MODEL    = "v_48_020"


# ── FoldMark-Protenix folding ─────────────────────────────────────────────────

def build_protenix_json(sequence: str, name: str) -> list[dict]:
    return [{"sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}], "name": name}]


def fold_with_protenix(
    sequence: str,
    name: str,
    out_dir: Path,
    watermark: bool,
    seed: int = 101,
    n_cycle: int = 10,
    n_sample: int = 1,
    n_step: int = 200,
) -> Path:
    """
    Fold `sequence`. `watermark=True` uses FoldMark-Protenix (message embedded
    in the backbone); `watermark=False` is the original predictor.

    Returns the path to the predicted CIF.
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

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(build_protenix_json(sequence, name), indent=2))

    arg_str = (
        f"--seeds {seed} "
        f"--dump_dir {out_dir} "
        f"--input_json_path {json_path} "
        f"--model.N_cycle {n_cycle} "
        f"--sample_diffusion.N_sample {n_sample} "
        f"--sample_diffusion.N_step {n_step}"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str, fill_required_with_null=True)
    configs.load_checkpoint_path = "./checkpoint.pt"
    download_infercence_cache()
    configs.input_json_path = update_infer_json(str(json_path), str(out_dir.parent), True)
    configs.watermark = watermark
    configs.saved_path = str(out_dir)

    runner = InferenceRunner(configs)
    infer_predict(runner, configs)

    cifs = sorted((out_dir / "predictions").glob("**/*.cif"))
    if not cifs:
        raise FileNotFoundError(f"No CIF produced under {out_dir}")
    return cifs[0]


def foldmark_detect_bits(cif_path: Path, n_bits: int = DEFAULT_N_CHUNKS) -> np.ndarray:
    """
    Recover the watermark bits from a FoldMark-predicted structure.

    Mirrors runner.inference.infer_detect:
        prediction['watermark'] -> clamp -> sigmoid -> threshold at 0.5
    """
    import torch
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, get_inference_dataloader

    arg_str = f"--input_json_path {cif_path} --dump_dir {cif_path.parent}"
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str, fill_required_with_null=True)
    configs.load_checkpoint_path = "./checkpoint.pt"

    runner = InferenceRunner(configs)
    dataloader = get_inference_dataloader(configs=configs)
    for batch in dataloader:
        prediction, _label = runner.detect(batch)
        code = torch.clamp(prediction["watermark"], min=-10, max=10)
        bits = (torch.sigmoid(code) > 0.5).long().cpu().numpy().ravel()
        return bits[:n_bits]
    raise RuntimeError(f"FoldMark detector produced no output for {cif_path}")


# ── scTM (self-consistency TM-score) ──────────────────────────────────────────

def cif_to_pdb(cif_path: Path, pdb_path: Path) -> Path:
    from Bio.PDB import MMCIFParser, PDBIO

    structure = MMCIFParser(QUIET=True).get_structure("s", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_path))
    return pdb_path


def extract_ca(pdb_path: Path, chain_id: str = "A") -> tuple[str, np.ndarray]:
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
    seq, coords = [], []
    for residue in structure[0][chain_id]:
        if "CA" not in residue:
            continue
        seq.append(seq1(residue.get_resname()))
        coords.append(residue["CA"].get_coord())
    return "".join(seq), np.asarray(coords, dtype=float)


def run_proteinmpnn(mpnn_dir: Path, pdb_path: Path, out_dir: Path, num_seq: int, seed: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(mpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", "A",
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(num_seq),
        "--sampling_temp", str(MPNN_TEMP),
        "--seed", str(seed),
        "--model_name", MPNN_MODEL,
        "--batch_size", "1",
        "--suppress_print", "1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mpnn_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(mpnn_dir), env=env, check=True)

    fa = out_dir / "seqs" / f"{pdb_path.stem}.fa"
    seqs, lines = [], fa.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">") and "sample=" in line and i + 1 < len(lines):
            seqs.append(lines[i + 1].strip())
    return seqs


_ESMFOLD = {}


def esmfold_pdb_string(sequence: str, device: str = "auto") -> str:
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
    return mdl.output_to_pdb(output)[0]


def sctm(
    backbone_pdb: Path,
    mpnn_dir: Path,
    work_dir: Path,
    seed: int,
    num_seq: int = N_MPNN_SEQS,
) -> float:
    """
    scTM = max over ProteinMPNN designs of TM-score(ESMFold(design), backbone).
    """
    from tmtools import tm_align

    ref_seq, ref_coords = extract_ca(backbone_pdb)
    designs = run_proteinmpnn(mpnn_dir, backbone_pdb, work_dir / "mpnn", num_seq, seed)

    best = 0.0
    for k, design in enumerate(designs):
        pdb_text = esmfold_pdb_string(design)
        tmp = work_dir / f"esmfold_{k}.pdb"
        tmp.write_text(pdb_text)
        mob_seq, mob_coords = extract_ca(tmp)
        if len(mob_coords) == 0:
            continue
        res = tm_align(ref_coords, mob_coords, ref_seq, mob_seq)
        best = max(best, float(max(res.tm_norm_chain1, res.tm_norm_chain2)))
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: fold both arms, score watermark recovery and scTM"
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--mpnn_dir", type=Path,
                    default=Path(os.environ.get("MPNN_DIR", "./ProteinMPNN")))
    ap.add_argument("--out_dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--work_dir", type=Path, default=None,
                    help="scratch dir for intermediate PDB/CIF (default: a temp dir)")
    ap.add_argument("--limit",    type=int, default=None,
                    help="only process the first N pairs (smoke test)")
    ap.add_argument("--n_chunks", type=int, default=DEFAULT_N_CHUNKS)
    ap.add_argument("--bits_per_chunk", type=int, default=1)
    ap.add_argument("--context_code_length", type=int, default=5)
    ap.add_argument("--vocab_size", type=int, default=32)
    ap.add_argument("--skip_sctm", action="store_true",
                    help="score watermark recovery only (much faster)")
    args = ap.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"Step 1 manifest not found: {args.manifest}\n"
            "Run step1_progen2_generation.py first."
        )
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[step2] {len(df)} matched pairs from {args.manifest}")

    cfg = ChenMultiBitConfig(
        n_chunks=args.n_chunks,
        bits_per_chunk=args.bits_per_chunk,
        context_code_length=args.context_code_length,
        vocab_size=args.vocab_size,
    )
    watermarker = ChenMultiBitWatermarker(cfg)

    tmp_ctx = tempfile.TemporaryDirectory() if args.work_dir is None else None
    work_root = args.work_dir or Path(tmp_ctx.name)
    work_root.mkdir(parents=True, exist_ok=True)

    per_seq: list[dict] = []
    paired:  list[dict] = []

    for idx, row in df.iterrows():
        sid    = row["seq_id"]
        length = int(row["length"])
        seed   = int(row["seed"])
        true_bits = np.array([int(b) for b in str(row["message"])], dtype=int)
        work = work_root / sid
        work.mkdir(parents=True, exist_ok=True)

        # ── chen arm: sequence already watermarked; fold WITHOUT FoldMark ─────
        seq_chen = str(row["sequence_chen"])
        token_ids = np.array([int(t) for t in str(row["token_ids_chen"]).split()], dtype=np.int64)
        acc_chen = watermarker.bit_accuracy(true_bits, token_ids[None, :])

        sctm_chen = float("nan")
        if not args.skip_sctm:
            cif = fold_with_protenix(seq_chen, f"{sid}_chen", work / "chen", watermark=False, seed=seed)
            pdb = cif_to_pdb(cif, work / "chen" / f"{sid}_chen.pdb")
            sctm_chen = sctm(pdb, args.mpnn_dir.resolve(), work / "chen_sctm", seed)

        # ── foldmark arm: clean sequence; fold WITH FoldMark ──────────────────
        seq_fm = str(row["sequence_foldmark"])
        sctm_fm = float("nan")
        acc_fm  = float("nan")
        if not args.skip_sctm:
            cif_fm = fold_with_protenix(seq_fm, f"{sid}_fm", work / "foldmark", watermark=True, seed=seed)
            pdb_fm = cif_to_pdb(cif_fm, work / "foldmark" / f"{sid}_fm.pdb")
            sctm_fm = sctm(pdb_fm, args.mpnn_dir.resolve(), work / "fm_sctm", seed)
            pred_bits = foldmark_detect_bits(cif_fm, n_bits=cfg.n_bits)
            n = min(pred_bits.size, true_bits.size)
            acc_fm = float((pred_bits[:n] == true_bits[:n]).mean())

        per_seq.append({"seq_id": sid, "arm": "chen", "length": length,
                        "bit_accuracy": acc_chen, "sctm": sctm_chen})
        per_seq.append({"seq_id": sid, "arm": "foldmark", "length": length,
                        "bit_accuracy": acc_fm, "sctm": sctm_fm})
        paired.append({"seq_id": sid, "length": length,
                       "bit_accuracy_chen": acc_chen, "bit_accuracy_foldmark": acc_fm,
                       "sctm_chen": sctm_chen, "sctm_foldmark": sctm_fm})

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(df)}] {sid}  "
                  f"acc chen={acc_chen:.3f} fm={acc_fm:.3f}  "
                  f"scTM chen={sctm_chen:.3f} fm={sctm_fm:.3f}")

    pd.DataFrame(per_seq).to_csv(args.out_dir / "per_sequence_scores.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out_dir / "paired_scores.csv", index=False)

    print(
        f"\n[step2] Done.\n"
        f"  Per-sequence → {args.out_dir/'per_sequence_scores.csv'}\n"
        f"  Paired       → {args.out_dir/'paired_scores.csv'}\n"
        f"\n  Pass to Step 3:  --paired_csv {args.out_dir/'paired_scores.csv'}\n"
    )
    if tmp_ctx is not None:
        tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
