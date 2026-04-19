#!/usr/bin/env python3
"""
Step 2 — Partial Inverse Folding with ProteinMPNN for eGFP
===========================================================

Runs ProteinMPNN on the FoldMark-watermarked eGFP backbone (from Step 1),
redesigning only the two surface-exposed loop regions while fixing all other
residues to their wild-type identity.

Design regions (1-indexed, chain A of 4EUL / watermarked structure):
    Region 1: residues 15 – 40   (N-terminal strand–loop–strand)
    Region 2: residues 160 – 190 (C-terminal barrel loops)

All other 172 positions are fixed → only the watermarked backbone segments
are sequence-redesigned.

Input  : tutorials/egfp/outputs/step1/egfp_watermarked.pdb
Output : tutorials/egfp/outputs/step2/
           seqs/egfp_watermarked.fa   ← raw MPNN FASTA
           mpnn_sequences.fasta       ← merged / annotated FASTA
           mpnn_sequences.csv         ← per-sequence stats (n_mut, recovery, …)

Run from the foldmark_space root:
    python tutorials/egfp/step2_proteinmpnn_inverse_folding.py \\
           --mpnn_dir ./ProteinMPNN

Dependencies:
    git clone https://github.com/dauparas/ProteinMPNN
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

# ── eGFP-specific constants ───────────────────────────────────────────────────

CHAIN = "A"

# Design regions: residues that MPNN is free to mutate (1-indexed, inclusive).
# Rationale: surface-exposed loops remote from the Thr65-Tyr66-Gly67 chromophore.
DESIGN_REGIONS: list[tuple[int, int]] = [
    (15, 40),    # N-terminal strand / loops
    (160, 190),  # C-terminal barrel loops
]

# ProteinMPNN settings used in the manuscript
MPNN_MODEL   = "v_48_020"
TEMPERATURE  = 0.1    # low temperature → high sequence recovery
N_SEQUENCES  = 100    # sequences generated per backbone
MPNN_SEED    = 42

DEFAULT_WM_PDB = Path("tutorials/egfp/outputs/step1/egfp_watermarked.pdb")
OUT_DIR        = Path("tutorials/egfp/outputs/step2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_chain_sequence(pdb_path: Path, chain: str) -> tuple[str, list[int]]:
    """
    Extract the CA-atom sequence and residue numbers for a given chain.
    Returns (sequence_string, list_of_residue_numbers_1based).
    """
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
    seq = "".join(seen[r] for r in resnums)
    return seq, resnums


def design_positions_from_regions(
    resnums: list[int], regions: list[tuple[int, int]]
) -> list[int]:
    """
    Return a sorted list of 1-based residue numbers that fall inside any design region.
    resnums: ordered PDB residue numbers for the chain.
    regions: list of (start, end) inclusive intervals in PDB numbering.
    """
    design_set: set[int] = set()
    for start, end in regions:
        design_set.update(r for r in resnums if start <= r <= end)
    return sorted(design_set)


def fixed_positions_for_mpnn(
    resnums: list[int], design_resnums: list[int]
) -> list[int]:
    """
    ProteinMPNN fixed_positions_jsonl uses 1-based *chain-local* indices
    (position of each residue in the MPNN chain array, starting at 1).
    Returns list of chain-local indices for positions NOT in design set.
    """
    design_set = set(design_resnums)
    fixed_chain_local = [
        chain_idx
        for chain_idx, resnum in enumerate(resnums, start=1)
        if resnum not in design_set
    ]
    return fixed_chain_local


def write_fixed_positions_jsonl(path: Path, pdb_key: str, chain: str, fixed: list[int]) -> None:
    path.write_text(json.dumps({pdb_key: {chain: fixed}}) + "\n")


def run_protein_mpnn(
    mpnn_dir: Path,
    pdb_path: Path,
    fixed_jsonl: Path,
    out_dir: Path,
    num_seq: int,
    temperature: float,
    seed: int,
    model_name: str,
    chain: str = "A",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(mpnn_dir / "protein_mpnn_run.py"),
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
    """Parse MPNN-designed sequences (headers start with 'T=' or contain 'sample=')."""
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


def apply_fixed_positions(
    mpnn_seq: str, wt_seq: str, design_resnums: list[int], resnums: list[int]
) -> str:
    """
    Build the merged sequence: MPNN amino acids at design positions,
    wild-type amino acids everywhere else.
    """
    design_set = set(design_resnums)
    mpnn_list = list(mpnn_seq)
    merged = []
    for chain_idx, resnum in enumerate(resnums):
        if resnum in design_set:
            merged.append(mpnn_list[chain_idx])
        else:
            merged.append(wt_seq[chain_idx])
    return "".join(merged)


def mutations_vs_reference(ref: str, var: str) -> tuple[list[int], str, float]:
    """Return (1-based mutation positions, detail string, recovery fraction)."""
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
        description="Step 2: ProteinMPNN partial inverse folding for eGFP"
    )
    ap.add_argument(
        "--watermarked_pdb",
        type=Path,
        default=DEFAULT_WM_PDB,
        help="Watermarked backbone PDB from Step 1.",
    )
    ap.add_argument(
        "--mpnn_dir",
        type=Path,
        default=Path(os.environ.get("MPNN_DIR", "./ProteinMPNN")),
        help="Path to cloned ProteinMPNN repository.",
    )
    ap.add_argument("--num_seq",    type=int,   default=N_SEQUENCES, help="Sequences to generate (default 100).")
    ap.add_argument("--temperature",type=float, default=TEMPERATURE,  help="Sampling temperature (default 0.1).")
    ap.add_argument("--seed",       type=int,   default=MPNN_SEED)
    ap.add_argument("--model_name", type=str,   default=MPNN_MODEL)
    ap.add_argument("--out_dir",    type=Path,  default=OUT_DIR)
    ap.add_argument("--skip_mpnn",  action="store_true", help="Reuse existing MPNN FASTA.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = args.watermarked_pdb.resolve()

    # ── 1. Extract sequence and residue numbers from watermarked PDB ──────────
    print(f"[step2] Reading backbone: {pdb_path}")
    wt_seq, resnums = read_chain_sequence(pdb_path, CHAIN)
    L = len(resnums)
    print(f"[step2] Chain {CHAIN}: {L} residues")

    # ── 2. Determine design vs. fixed positions ───────────────────────────────
    design_resnums = design_positions_from_regions(resnums, DESIGN_REGIONS)
    fixed_chain_local = fixed_positions_for_mpnn(resnums, design_resnums)
    n_design = len(design_resnums)
    n_fixed  = len(fixed_chain_local)

    print(f"[step2] Design regions : {DESIGN_REGIONS}")
    print(f"[step2] Design residues: {n_design}  (residues {design_resnums[:5]}…)")
    print(f"[step2] Fixed residues : {n_fixed} / {L}")

    # ── 3. Write fixed-positions JSONL ────────────────────────────────────────
    pdb_key      = pdb_path.stem
    fixed_jsonl  = args.out_dir / "fixed_positions.jsonl"
    write_fixed_positions_jsonl(fixed_jsonl, pdb_key, CHAIN, fixed_chain_local)

    # ── 4. Run ProteinMPNN ────────────────────────────────────────────────────
    if not args.skip_mpnn:
        mpnn_dir = args.mpnn_dir.resolve()
        if not (mpnn_dir / "protein_mpnn_run.py").is_file():
            sys.exit(
                f"[error] protein_mpnn_run.py not found in {mpnn_dir}\n"
                "        Clone ProteinMPNN: git clone https://github.com/dauparas/ProteinMPNN"
            )
        run_protein_mpnn(
            mpnn_dir=mpnn_dir,
            pdb_path=pdb_path,
            fixed_jsonl=fixed_jsonl,
            out_dir=args.out_dir,
            num_seq=args.num_seq,
            temperature=args.temperature,
            seed=args.seed,
            model_name=args.model_name,
            chain=CHAIN,
        )

    # ── 5. Parse MPNN output ──────────────────────────────────────────────────
    fa_path = args.out_dir / "seqs" / f"{pdb_key}.fa"
    if not fa_path.is_file():
        sys.exit(f"[error] MPNN output not found: {fa_path}")
    samples = parse_mpnn_fasta(fa_path)
    print(f"[step2] Parsed {len(samples)} sequences from MPNN.")

    # ── 6. Merge fixed positions + compute stats ──────────────────────────────
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
            "esm2_650M_pll_mean": "TBD",  # filled in Step 3
        })

    # Summary
    recoveries = [r["sequence_recovery"] for r in rows]
    print(
        f"\n--- Sequence recovery ({len(rows)} sequences) ---\n"
        f"  Mean   : {np.mean(recoveries):.1%}\n"
        f"  Median : {np.median(recoveries):.1%}\n"
        f"  SD     : {np.std(recoveries):.1%}\n"
        f"  Min    : {np.min(recoveries):.1%}\n"
        f"  Max    : {np.max(recoveries):.1%}\n"
        f"  Design residues only: "
        f"{n_design} positions out of {L} were re-designed\n"
    )

    # ── 7. Write CSV ──────────────────────────────────────────────────────────
    csv_path = args.out_dir / "mpnn_sequences.csv"
    fieldnames = [
        "sample_id", "n_mutations", "sequence_recovery",
        "esm2_650M_pll_mean",
        "mutation_positions", "mutations_detail", "sequence", "mpnn_header",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ── 8. Write annotated FASTA ──────────────────────────────────────────────
    fasta_path = args.out_dir / "mpnn_sequences.fasta"
    with fasta_path.open("w") as f:
        # Wild-type reference first
        f.write(f">wildtype|n_mut=0|recovery=1.0\n{wt_seq}\n")
        for row in rows:
            f.write(
                f">sample_{row['sample_id']:03d}"
                f"|n_mut={row['n_mutations']}"
                f"|recovery={row['sequence_recovery']:.4f}"
                f"|{row['mpnn_header']}\n"
                f"{row['sequence']}\n"
            )

    print(f"[step2] CSV  → {csv_path}")
    print(f"[step2] FASTA→ {fasta_path}")
    print(f"\n[step2] Done. Pass to Step 3:\n  --mpnn_csv {csv_path}")


if __name__ == "__main__":
    main()
