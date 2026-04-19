# eGFP Tutorial (PDB 4EUL)

## Structure choice

**PDB 4EUL** — enhanced GFP (S65T / H148D), active fluorescent state, 1.35 Å,
chain A, 239 residues.

The **active (chromophore-containing) conformation** was chosen because:
1. The fluorescence assay directly measures the mature chromophore state, so the
   design template must match the assayed form.
2. The chromophore geometry (Thr65-Tyr66-Gly67) is explicitly resolved at high
   resolution, allowing those residues to be pinned during MPNN inverse folding.
3. ESMFold and AF3 predict this fold for EGFP sequences; FoldMark's watermarked
   Protenix output is therefore directly comparable to 4EUL.

## MPNN design regions

Only two surface-exposed loop/strand segments are redesigned; all other positions
are fixed to wild-type identity:

| Region | Residues (1-indexed, chain A) | Secondary structure context |
|--------|-------------------------------|-----------------------------|
| Loop 1 | 15 – 40  | N-terminal strand–loop–strand |
| Loop 2 | 160 – 190 | C-terminal barrel loops |

These regions were selected because:
- They are solvent-exposed and remote from the chromophore pocket.
- The FoldMark watermark subtly perturbs the backbone in these segments.
- ProteinMPNN can reliably redesign them at T = 0.1 with high recovery (~87%).

## Running the three steps

```bash
# from the foldmark_space root directory
python tutorials/egfp/step1_watermarked_structure_prediction.py
python tutorials/egfp/step2_proteinmpnn_inverse_folding.py \
       --mpnn_dir ./ProteinMPNN
python tutorials/egfp/step3_esm2_ranking.py
```

Outputs land in `tutorials/egfp/outputs/`.

## Example designs

Three representative constructs shown in the manuscript figure (panel c):

| Construct | pLDDT | TM-score vs WT | Watermark detected |
|-----------|-------|----------------|--------------------|
| FoldMark_EGFP_2 | 94.00 | 0.9814 | True |
| FoldMark_EGFP_6 | 93.57 | 0.9805 | True |
| FoldMark_EGFP_8 | 94.10 | 0.9833 | True |

TM-scores > 0.98 confirm that watermarking introduces only minimal structural
deviation while remaining detectable.

<div align=center>
<img src="../assets/egfp_cas13.png" width="700"/>
</div>

## Wet-lab results

12 constructs (top-ranked by ESM2-650M PLL, ≤ 25 mutations) were synthesised
and assayed:
- 12/12 fluorescent; mean MFI = 98% of wild-type.
- Watermark bit accuracy > 90% on all tested sequences.
