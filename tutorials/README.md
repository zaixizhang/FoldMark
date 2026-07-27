# FoldMark Wet-Lab Reproduction Tutorials

This directory contains step-by-step scripts to reproduce the eGFP, Cas13, DHFR
and TEV protease watermarking experiments described in the FoldMark manuscript.

## Directory layout

```
tutorials/
├── README.md          ← this file
├── egfp/              ← eGFP (PDB 4EUL) tutorial
│   ├── README.md
│   ├── step1_watermarked_structure_prediction.py
│   ├── step2_proteinmpnn_inverse_folding.py
│   └── step3_esm2_ranking.py
├── cas13/             ← Cas13 (PDB 7VTI) tutorial
│   ├── README.md
│   ├── step1_watermarked_structure_prediction.py
│   ├── step2_proteinmpnn_inverse_folding.py
│   └── step3_esm2_ranking.py
├── dhfr/              ← E. coli DHFR (PDB 4P66) tutorial
│   ├── README.md
│   ├── step1_watermarked_structure_prediction.py
│   ├── step2_proteinmpnn_inverse_folding.py
│   └── step3_esm2_ranking.py
└── tev/               ← TEV protease (PDB 1LVM) tutorial
    ├── README.md
    ├── step1_watermarked_structure_prediction.py
    ├── step2_proteinmpnn_inverse_folding.py
    └── step3_esm2_ranking.py
```

## Targets

| Target | PDB | Length | Design region(s) | Functional assay |
|--------|-----|--------|------------------|------------------|
| eGFP | 4EUL | 239 aa | 15–40, 160–190 (surface loops) | Fluorescence / MFI |
| Cas13 | 7VTI | ~800 aa | 258–325 (lid) | SHERLOCK editing |
| DHFR | 4P66 | 159 aa | 150–157 (C-terminal tail) | DHF → THF reductase activity |
| TEV | 1LVM | 241 aa | group A: 11–25 + alternating 178–197; group B: 11–25 + sparse 222–241 | ENLYFQ↓G cleavage |

The two enzymes (DHFR, TEV) are the hardest cases: both are small, compact and
have no spare surface loops, so their watermark carriers are narrow and — for
TEV — **sparse** (alternating or 1-in-4 positions) rather than solid blocks.
Catalytic residues are held fixed by an explicit hold list in each Step 2 script.

## Three-step pipeline

```
Step 1 ── FoldMark-Protenix structure prediction (watermark embedded)
              ↓  watermarked backbone  (.cif → .pdb)
Step 2 ── ProteinMPNN partial inverse folding
              ↓  100 candidate sequences  (.fasta)
Step 3 ── ESM2-650M pseudo-log-likelihood ranking
              ↓  top-ranked sequences ready for synthesis
```

## FAQ

| Question | Answer |
|----------|--------|
| How many MPNN runs? | 100 sequences per backbone (`--num_seq 100`, T = 0.1) |
| Entire protein or specific regions? | Partial: only surface-exposed, non-functional loops redesigned (see per-protein README) |
| Sequence recovery? | eGFP ~87%, Cas13 ~85%, DHFR ~96%, TEV ~95% overall (see Step 3 CSV) |
| Which eGFP conformation? | 4EUL — active/fluorescent state with mature chromophore |
| Which Cas13 conformation? | 7VTI — apo (inactive) state; rationale in cas13/README.md |
| How are catalytic residues protected? | Each Step 2 script carries an explicit hold set — DHFR: Met20 loop, Asp27, folate cleft; TEV: His46/Asp81/Cys151 triad, Trp198, Cys19 |
| Why do the enzymes use narrower windows? | DHFR (159 aa) and TEV (241 aa) are compact single/two-domain enzymes with no disposable loops; the carrier is kept small and, for TEV, sparse |
| Why is TEV's TM-score vs 1LVM only ~0.94? | The 241-aa construct includes a 20-residue engineered C-terminal extension absent from the crystal. Wild type scores the same 0.94 — see tev/README.md |

## Installation

```bash
# 1. Clone the FoldMark HuggingFace Space (contains FoldMark-Protenix)
git clone https://huggingface.co/spaces/Zaixi/FoldMark foldmark_space
cd foldmark_space

# 2. Install dependencies
pip install -r requirements.txt

# 3. Clone ProteinMPNN
git clone https://github.com/dauparas/ProteinMPNN

# 4. ESM2 for ranking
pip install fair-esm

# 5. Run tutorials from the foldmark_space root, e.g.:
python tutorials/egfp/step1_watermarked_structure_prediction.py
python tutorials/egfp/step2_proteinmpnn_inverse_folding.py
python tutorials/egfp/step3_esm2_ranking.py

# The enzymes follow the same three steps; TEV takes a carrier group:
python tutorials/dhfr/step2_proteinmpnn_inverse_folding.py --mpnn_dir ./ProteinMPNN
python tutorials/tev/step2_proteinmpnn_inverse_folding.py  --mpnn_dir ./ProteinMPNN --group a
```

## Supplementary experiments

The tutorials above reproduce the wet-lab constructs. The additional analyses
accompanying the manuscript — baseline comparison, robustness to structural
post-processing, million-user tracing, and the inverse-fold/re-fold round trip —
live in [`experiments_revision/`](../experiments_revision/) and follow the same
numbered `stepN_*.py` layout.
