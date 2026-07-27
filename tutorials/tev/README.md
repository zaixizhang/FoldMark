# TEV Protease Tutorial (PDB 1LVM)

## Structure choice

**PDB 1LVM** — Tobacco Etch Virus NIa protease (TEV), catalytic domain,
chain A, residues **1 – 221**.

The construct used here is **241 aa**: the 221-residue crystallographic core
plus a **20-residue engineered C-terminal extension (222 – 241)** that is not
present in the crystal. Numbering is 1-indexed with position 1 = the G of
"GES…", which matches 1LVM chain A numbering directly.

| Segment | Residues | Note |
|---------|----------|------|
| TEV NIa catalytic domain | 1 – 221 | Corresponds to 1LVM chain A |
| Engineered C-terminal extension | 222 – 241 | Absent from the crystal; furthest from the active site |

Known difference of this construct vs. the crystal: **Q74H**.

Because group B below places watermark bits in the 222 – 241 extension, Step 1
predicts the **full-length 241-aa construct** rather than reusing the crystal —
the crystal simply does not contain those coordinates.

## MPNN design regions

TEV is a compact **two-β-barrel** chymotrypsin-like protease with no large
disposable loops, so the watermark is carried on **sparse** positions rather
than a solid redesigned block. Two groups are provided:

### Group A (default) — carrier stays inside the crystallographic core

| Region | Residues | Scheme | Structural context |
|--------|----------|--------|---------------------|
| β1 N-terminal loop | 11 – 25 | all designable **except Cys19** | Core-internal but relatively tolerant loop |
| β2 C-terminal helix periphery | 178 – 197 | **alternating** — 178, 180, …, 196 designable | Outer helix bundle |

- **Cys19** is a structural cysteine and is always held fixed.
- **197 is a hard right edge**: Trp198 begins the substrate-tunnel region.

### Group B — carrier uses the engineered extension

| Region | Residues | Scheme | Structural context |
|--------|----------|--------|---------------------|
| β1 N-terminal loop | 11 – 25 | as group A | — |
| Engineered C-terminal extension | 222 – 241 | **1-in-4** — 222, 226, 230, 234, 238 designable | Outside the crystallographic core; lowest functional risk |

### Regions that are never watermarked

| Residue | Role |
|---------|------|
| **His46** | Catalytic triad |
| **Asp81** | Catalytic triad |
| **Cys151** | Catalytic nucleophile |
| **Trp198** | Substrate tunnel |
| **Cys19** | Structural cysteine inside the 11 – 25 window |

These are enforced by the `CATALYTIC_HOLD` / `STRUCTURAL_CYS` sets in
`step2_proteinmpnn_inverse_folding.py`.

## Running the three steps

```bash
# from the foldmark_space root directory
python tutorials/tev/step1_watermarked_structure_prediction.py
python tutorials/tev/step2_proteinmpnn_inverse_folding.py \
       --mpnn_dir ./ProteinMPNN --group a
python tutorials/tev/step3_esm2_ranking.py --group a

# group B (C-terminal extension carrier)
python tutorials/tev/step2_proteinmpnn_inverse_folding.py \
       --mpnn_dir ./ProteinMPNN --group b
python tutorials/tev/step3_esm2_ranking.py --group b
```

Outputs land in `tutorials/tev/outputs/`.

## Example designs

Top-ranked constructs per group (ESM2-650M PLL; TM-score computed against the
1LVM crystal structure after ESMFold re-prediction of the 241-aa construct):

| Construct | Group | n_mut | ESM2-650M PLL | TM-score vs 1LVM | TM-score vs WT fold |
|-----------|-------|-------|---------------|------------------|---------------------|
| wild type | — | 0 | −1.4490 | 0.9417 | 1.0000 |
| FoldMark_TEV_A1 | A | 10 | −1.4926 | 0.9391 | 0.9854 |
| FoldMark_TEV_A2 | A | 11 | −1.4936 | 0.9429 | 0.9912 |
| FoldMark_TEV_B1 | B | 11 | −1.5061 | 0.9444 | 0.9694 |
| FoldMark_TEV_B2 | B | 11 | −1.5068 | 0.9421 | 0.9736 |

The TM-score vs. the 1LVM crystal is ~0.94 for **all** entries **including the
wild type** (0.9417). That ceiling is set by the 20-residue engineered extension,
which is disordered/absent in the crystal and therefore cannot superimpose — it
is not a watermarking artefact. The meaningful comparison is TM-score vs. the
wild-type predicted fold, which stays **> 0.96** for every design.

## Wet-lab results

Top-ranked constructs were expressed and assayed for **ENLYFQ↓G cleavage
efficiency** against a standard substrate:

- **2 out of 3 top-ranked variants retained activity equivalent to the wild-type
  enzyme.**
- Watermark bit accuracy > 90 % on the tested sequences.
