#!/usr/bin/env python3
"""
Step 2 — Partial Inverse Folding with ProteinMPNN for Cas13 (PDB 7VTI)
=======================================================================

Runs ProteinMPNN on the FoldMark-watermarked Cas13 backbone (from Step 1),
redesigning only the lid region while fixing all other residues to wild-type.

Design region (1-indexed, chain A of 7VTI / watermarked structure):
    Lid region: residues 258 – 325

The lid region is a helical domain that covers the RNA-binding cleft in the
apo state.  It is surface-exposed, remote from the HEPN catalytic dyads, and
is the locus of the strongest FoldMark backbone perturbation.  All other
residues — including the HEPN1 (Arg116 / His120) and HEPN2 (Arg788 / His792)
catalytic residues — are fixed to wild-type identity.

Input  : tutorials/cas13/outputs/step1/cas13_watermarked.pdb
Output : tutorials/cas13/outputs/step2/
           seqs/cas13_watermarked.fa   ← raw MPNN FASTA
           mpnn_sequences.fasta        ← merged / annotated FASTA
           mpnn_sequences.csv          ← per-sequence stats

Run from the foldmark_space root:
    python tutorials/cas13/step2_proteinmpnn_inverse_folding.py \\
           --mpnn_dir ./ProteinMPNN
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── Cas13-specific constants ──────────────────────────────────────────────────

CHAIN = "A"

# Only the lid region is redesigned.
DESIGN_REGIONS: list[tuple[int, int]] = [
    (258, 325),   # helical lid domain covering the RNA-binding cleft
]

# ProteinMPNN settings (manuscript values)
MPNN_MODEL   = "v_48_020"
TEMPERATURE  = 0.1
N_SEQUENCES  = 100
MPNN_SEED    = 42

DEFAULT_WM_PDB = Path("tutorials/cas13/outputs/step1/cas13_watermarked.pdb")
OUT_DIR        = Path("tutorials/cas13/outputs/step2")


# ── Helpers (identical logic to egfp/step2) ───────────────────────────────────

def read_chain_sequence(pdb_path: Path, chain: str) -> tuple[str, list[int]]:
    three_to_one = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
        "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
        "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
        "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    seen: dict[int, str] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            resname = line[17:20].strip()
            resnum  = int(line[22:26].strip())
            aa = three_to_one.get(resname)
            if aa and resnum not in seen:
                seen[resnum] = aa
    if not seen:
        raise ValueError(f"No CA atoms for chain {chain} in {pdb_path}")
    resnums = sorted(seen)
    return "".join(seen[r] for r in resnums), resnums


def design_positions_from_regions(resnums: list[int], regions: list[tuple[int, int]]) -> list[int]:
    design_set: set[int] = set()
    for start, end in regions:
        design_set.update(r for r in resnums if start <= r <= end)
    return sorted(design_set)


def fixed_positions_for_mpnn(resnums: list[int], design_resnums: list[int]) -> list[int]:
    design_set = set(design_resnums)
    return [ci for ci, rn in enumerate(resnums, start=1) if rn not in design_set]


def write_fixed_positions_jsonl(path: Path, pdb_key: str, chain: str, fixed: list[int]) -> None:
    path.write_text(json.dumps({pdb_key: {chain: fixed}}) + "\n")


def run_protein_mpnn(
    mpnn_dir: Path, pdb_path: Path, fixed_jsonl: Path, out_dir: Path,
    num_seq: int, temperature: float, seed: int, model_name: str, chain: str = "A",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(mpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path",              str(pdb_path),
        "--pdb_path_chains",       chain,
        "--fixed_positions_jsonl", str(fixed_jsonl),
        "--out_folder",            str(out_dir),
        "--num_seq_per_target",    str(num_seq),
        "--sampling_temp",         str(temperature),
        "--seed",                  str(seed),
        "--model_name",            model_name,
        "--batch_size",            "1",
        "--suppress_print",        "1",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mpnn_dir) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[step2] ProteinMPNN: {num_seq} seqs, T={temperature}, model={model_name}")
    subprocess.run(cmd, cwd=str(mpnn_dir), env=env, check=True)
    print("[step2] ProteinMPNN done.")


def parse_mpnn_fasta(fa_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    lines = fa_path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(">T=") or (line.startswith(">") and "sample=" in line):
            hdr = line[1:]
            seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            entries.append((hdr, seq))
            i += 2
        else:
            i += 1
    return entries


def apply_fixed_positions(mpnn_seq: str, wt_seq: str, design_resnums: list[int], resnums: list[int]) -> str:
    design_set = set(design_resnums)
    return "".join(
        mpnn_seq[ci] if rn in design_set else wt_seq[ci]
        for ci, rn in enumerate(resnums)
    )


def mutations_vs_reference(ref: str, var: str) -> tuple[list[int], str, float]:
    assert len(ref) == len(var)
    pos, parts = [], []
    for i, (a, b) in enumerate(zip(ref, var)):
        if a != b:
            pos.append(i + 1)
            parts.append(f"{i+1}{a}>{b}")
    return pos, ",".join(parts), 1.0 - len(pos) / len(ref)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: ProteinMPNN partial inverse folding for Cas13 (lid region 258-325)"
    )
    ap.add_argument("--watermarked_pdb", type=Path, default=DEFAULT_WM_PDB)
    ap.add_argument(
        "--mpnn_dir",
        type=Path,
        default=Path(os.environ.get("MPNN_DIR", "./ProteinMPNN")),
    )
    ap.add_argument("--num_seq",    type=int,   default=N_SEQUENCES)
    ap.add_argument("--temperature",type=float, default=TEMPERATURE)
    ap.add_argument("--seed",       type=int,   default=MPNN_SEED)
    ap.add_argument("--model_name", type=str,   default=MPNN_MODEL)
    ap.add_argument("--out_dir",    type=Path,  default=OUT_DIR)
    ap.add_argument("--skip_mpnn",  action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = args.watermarked_pdb.resolve()

    # ── 1. Extract sequence ───────────────────────────────────────────────────
    print(f"[step2] Reading backbone: {pdb_path}")
    wt_seq, resnums = read_chain_sequence(pdb_path, CHAIN)
    L = len(resnums)
    print(f"[step2] Chain {CHAIN}: {L} residues")

    # ── 2. Design vs. fixed positions ─────────────────────────────────────────
    design_resnums   = design_positions_from_regions(resnums, DESIGN_REGIONS)
    fixed_chain_local = fixed_positions_for_mpnn(resnums, design_resnums)
    n_design = len(design_resnums)

    print(f"[step2] Design region  : {DESIGN_REGIONS}  (lid, {n_design} residues)")
    print(f"[step2] Fixed residues : {L - n_design} / {L}")
    print(f"[step2] NOTE: HEPN catalytic dyads are fixed by default "
          f"(they fall outside the lid region 258–325).")

    # ── 3. Fixed-positions JSONL ──────────────────────────────────────────────
    pdb_key     = pdb_path.stem
    fixed_jsonl = args.out_dir / "fixed_positions.jsonl"
    write_fixed_positions_jsonl(fixed_jsonl, pdb_key, CHAIN, fixed_chain_local)

    # ── 4. Run ProteinMPNN ────────────────────────────────────────────────────
    if not args.skip_mpnn:
        mpnn_dir = args.mpnn_dir.resolve()
        if not (mpnn_dir / "protein_mpnn_run.py").is_file():
            sys.exit(
                f"[error] ProteinMPNN not found at {mpnn_dir}\n"
                "        git clone https://github.com/dauparas/ProteinMPNN"
            )
        run_protein_mpnn(
            mpnn_dir=mpnn_dir, pdb_path=pdb_path, fixed_jsonl=fixed_jsonl,
            out_dir=args.out_dir, num_seq=args.num_seq, temperature=args.temperature,
            seed=args.seed, model_name=args.model_name, chain=CHAIN,
        )

    # ── 5. Parse + merge ──────────────────────────────────────────────────────
    fa_path = args.out_dir / "seqs" / f"{pdb_key}.fa"
    if not fa_path.is_file():
        sys.exit(f"[error] MPNN output not found: {fa_path}")
    samples = parse_mpnn_fasta(fa_path)
    print(f"[step2] Parsed {len(samples)} sequences.")

    rows: list[dict] = []
    for idx, (mpnn_hdr, mpnn_seq) in enumerate(samples, start=1):
        merged = apply_fixed_positions(mpnn_seq, wt_seq, design_resnums, resnums)
        mut_pos, mut_detail, recovery = mutations_vs_reference(wt_seq, merged)
        rows.append({
            "sample_id"         : idx,
            "mpnn_header"       : mpnn_hdr,
            "sequence"          : merged,
            "n_mutations"       : len(mut_pos),
            "sequence_recovery" : recovery,
            "mutation_positions": ",".join(str(p) for p in mut_pos),
            "mutations_detail"  : mut_detail,
            "esm2_650M_pll_mean": "TBD",
        })

    recoveries = [r["sequence_recovery"] for r in rows]
    print(
        f"\n--- Sequence recovery ({len(rows)} sequences) ---\n"
        f"  Mean   : {np.mean(recoveries):.1%}\n"
        f"  Median : {np.median(recoveries):.1%}\n"
        f"  SD     : {np.std(recoveries):.1%}\n"
        f"  Min    : {np.min(recoveries):.1%}\n"
        f"  Max    : {np.max(recoveries):.1%}\n"
        f"  Lid design region: {n_design} residues redesigned / {L} total\n"
    )

    # ── 6. CSV ────────────────────────────────────────────────────────────────
    csv_path = args.out_dir / "mpnn_sequences.csv"
    fieldnames = [
        "sample_id","n_mutations","sequence_recovery","esm2_650M_pll_mean",
        "mutation_positions","mutations_detail","sequence","mpnn_header",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── 7. Annotated FASTA ────────────────────────────────────────────────────
    fasta_path = args.out_dir / "mpnn_sequences.fasta"
    with fasta_path.open("w") as f:
        f.write(f">wildtype_Cas13Bt3_7VTI|n_mut=0|recovery=1.0\n{wt_seq}\n")
        for row in rows:
            f.write(
                f">sample_{row['sample_id']:03d}"
                f"|n_mut={row['n_mutations']}"
                f"|recovery={row['sequence_recovery']:.4f}"
                f"|{row['mpnn_header']}\n"
                f"{row['sequence']}\n"
            )

    print(f"[step2] CSV   → {csv_path}")
    print(f"[step2] FASTA → {fasta_path}")
    print(f"\n[step2] Done. Pass to Step 3:\n  --mpnn_csv {csv_path}")


if __name__ == "__main__":
    main()
