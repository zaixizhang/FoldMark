# FoldMark Wet-Lab Reproduction Tutorials

This directory contains step-by-step scripts to reproduce the eGFP and Cas13
watermarking experiments described in the FoldMark manuscript.

## Directory layout

```
tutorials/
├── README.md          ← this file
├── egfp/              ← eGFP (PDB 4EUL) tutorial
│   ├── README.md
│   ├── step1_watermarked_structure_prediction.py
│   ├── step2_proteinmpnn_inverse_folding.py
│   └── step3_esm2_ranking.py
└── cas13/             ← Cas13 (PDB 7VTI) tutorial
    ├── README.md
    ├── step1_watermarked_structure_prediction.py
    ├── step2_proteinmpnn_inverse_folding.py
    └── step3_esm2_ranking.py
```

## Three-step pipeline

```
Step 1 ── FoldMark-Protenix structure prediction (watermark embedded)
              ↓  watermarked backbone  (.cif → .pdb)
Step 2 ── ProteinMPNN partial inverse folding
              ↓  100 candidate sequences  (.fasta)
Step 3 ── ESM2-650M pseudo-log-likelihood ranking
              ↓  top-ranked sequences ready for synthesis
```

## Reviewer Q&A

| Question | Answer |
|----------|--------|
| How many MPNN runs? | 100 sequences per backbone (`--num_seq 100`, T = 0.1) |
| Entire protein or specific regions? | Partial: only surface-exposed, non-functional loops redesigned (see per-protein README) |
| Sequence recovery? | eGFP ~87%, Cas13 ~85% overall (see Step 3 CSV) |
| Which eGFP conformation? | 4EUL — active/fluorescent state with mature chromophore |
| Which Cas13 conformation? | 7VTI — apo (inactive) state; rationale in cas13/README.md |

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
```
