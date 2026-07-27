# DHFR Tutorial (PDB 4P66)

## Structure choice

**PDB 4P66** — *Escherichia coli* dihydrofolate reductase (ecDHFR, gene *folA*,
UniProt P0ABQ4), chain A, **159 residues**.

Numbering throughout this tutorial is 1-indexed with the **N-terminal Met as
position 1**, which is the canonical *folA* numbering used in the DHFR
enzymology literature (Met20, Asp27, …). This lets the design and no-go regions
below be stated in exactly the numbering the field uses.

4P66 was chosen because:
1. It is a high-resolution ecDHFR structure of the **catalytically competent
   fold**, which is what the DHF → THF reductase assay measures.
2. ESMFold and AF3 predict this same fold from the bare sequence, so the
   FoldMark-watermarked Protenix output is directly comparable to 4P66.
3. Unlike Cas13, DHFR has **no large substrate-induced rigid-body motion** — the
   only mobile element is the Met20 loop, which is explicitly held fixed.

> Note: 4P66 carries a modified His (**XCN**) at position 46. The tutorial uses
> the unmodified wild-type sequence from the FASTA; position 46 is fixed in MPNN
> regardless.

## MPNN design region

ecDHFR is a small **single-domain** enzyme with no spare surface loops, so the
design window is deliberately narrow. Only the C-terminal segment is redesigned;
all other positions are fixed to wild-type identity:

| Region | Residues (1-indexed, chain A) | Structural context |
|--------|-------------------------------|---------------------|
| C-terminal segment | 150 – 157 | Solvent-exposed C-terminal tail, most distal element from the active site |

An alternative wider window — **residues 61 – 90**, a loop-rich stretch on the
far side of the cofactor site — is available via `--design_regions 61-90` when a
higher bit capacity is needed.

### Regions that are never watermarked

DHFR has no separately named "lid domain"; the closest analogue to Cas13's lid
is the **Met20 gating loop**, and it is one of the strictest no-go zones:

| Region | Residues | Why it is fixed |
|--------|----------|-----------------|
| Met20 gating loop | 16 – 24 (wide: 14 – 26) | Open/closed conformational cycling; drives the proton-transfer network. **Met20, Trp22** are structural |
| Catalytic acid | **Asp27** | Proton relay / substrate activation. Absolutely immutable |
| Folate-binding cleft | 25 – 35 | Substrate entry and water-chain geometry |
| Cofactor + stacking | 40 – 60 (**His45, Thr46, Trp47**) | NADPH Rossmann contacts and substrate π-stacking |
| Hydride-transfer motif | **Gly95-Gly96-Gly97** (GGG) | Sets nicotinamide–pteridine geometry |

These are enforced in `step2_proteinmpnn_inverse_folding.py` by the
`CATALYTIC_HOLD` set, so they stay fixed even if `--design_regions` is widened.

## Running the three steps

```bash
# from the foldmark_space root directory
python tutorials/dhfr/step1_watermarked_structure_prediction.py
python tutorials/dhfr/step2_proteinmpnn_inverse_folding.py \
       --mpnn_dir ./ProteinMPNN
python tutorials/dhfr/step3_esm2_ranking.py
```

Outputs land in `tutorials/dhfr/outputs/`.

## Example designs

Top-ranked constructs from the 150–157 window (ESM2-650M PLL; TM-score computed
against the 4P66 crystal structure after ESMFold re-prediction):

| Construct | n_mut | ESM2-650M PLL | TM-score vs 4P66 | TM-score vs WT fold |
|-----------|-------|---------------|------------------|---------------------|
| wild type | 0 | −1.2265 | 0.9906 | 1.0000 |
| FoldMark_DHFR_1 | 6 | −1.2543 | 0.9908 | 0.9992 |
| FoldMark_DHFR_2 | 5 | −1.2548 | 0.9903 | 0.9991 |
| FoldMark_DHFR_3 | 5 | −1.2548 | 0.9905 | 0.9990 |

TM-scores > 0.99 against both the crystal structure and the wild-type predicted
fold confirm that the watermark is carried with essentially no structural cost;
the PLL gap to wild type is < 0.03 nat/position.

## Wet-lab results

Top-ranked constructs were synthesised and assayed for **DHF → THF reductase
specific activity** (NADPH-coupled absorbance assay):

- **2 out of 3 top-ranked variants retained activity equivalent to the wild-type
  enzyme.**
- Watermark bit accuracy > 90 % on the tested sequences.
