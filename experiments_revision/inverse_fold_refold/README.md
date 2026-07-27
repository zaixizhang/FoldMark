# Watermark survival through inverse folding + clean re-folding

Does the FoldMark watermark survive a full round trip through **sequence space**?

The watermark lives in the backbone geometry. Inverse folding reduces a
structure to a sequence, which carries no explicit watermark bits; folding that
sequence again with a model that has never seen the watermarking module
regenerates a structure from scratch. If the mark is still recoverable
afterwards, it must have propagated into the residue choices ProteinMPNN made,
and then been re-expressed in the geometry a clean predictor produced.

This is the strongest form of the removal attack in the sense that the attacker
never touches the watermark at all — they simply re-derive the structure through
a channel with no knowledge of it.

```
watermarked structure (32-bit message embedded)
        │
Step 1  ├─► ProteinMPNN inverse folding ──► sequences
        │
Step 2  ├─► clean re-folding: Protenix (watermark=False) / ESMFold / AlphaFold3
        │       └─► TM-align back to the watermarked backbone
        │
Step 3  └─► FoldMark decoder on the re-folded structure ──► 32 bits ──► ROC
```

## Conditions

| Condition | Meaning |
|-----------|---------|
| `original` | The watermarked structure itself, no round trip (baseline) |
| `protenix` | Inverse fold → Protenix with `watermark=False` — same architecture, watermarking path disabled |
| `esmfold` | Inverse fold → ESMFold (`facebook/esmfold_v1`) — a completely different model |
| `af3` | Inverse fold → AlphaFold3, through an external command |

Every stage runs at defaults. Nothing in the chain is tuned to preserve the
mark, and the re-folders are genuinely clean — the point of the experiment is
lost if the re-folding model knows about FoldMark.

## Detection is unchanged: still 32 bits

Step 3 uses the ordinary FoldMark decoder on the re-folded structure, with no
re-training and no knowledge of which folder produced it:

```
prediction['watermark'] → clamp(-10, 10) → sigmoid → threshold at 0.5
```

The statistic is **k**, the number of the 32 bits matching the embedded message.
Under H0 the bits are fair coins, so

```
FPR(tau) = P[X >= tau],  X ~ Binomial(32, 0.5)
```

is analytic — which is what lets an FPR of 1e-5 be quoted without 10^5 negative
structures. **TPR is empirical**: the fraction of round-tripped structures whose
k reaches the threshold. Every plotted point is measured; only the FPR axis is
modelled.

## Reading the result: TM-score matters

Step 2 TM-aligns every re-folded structure back to the original watermarked
backbone, and Step 3 carries that number through to the summary table. This
separates two very different explanations for a low detection rate:

| Observation | Interpretation |
|-------------|----------------|
| High TM, low bit accuracy | The round trip genuinely **removed** the watermark |
| Low TM | The structure was **not reproduced at all**, so watermark loss says nothing about robustness |

Quoting a TPR without the accompanying TM-score is not interpretable, which is
why `tpr_at_fpr.csv` always carries `mean_tm_to_watermarked` next to it.

## Aggregating MPNN samples

Step 1 produces several sequences per structure. Step 3's `--aggregate` controls
how they are combined:

- `best` (default) — the sample whose re-fold yields the most matching bits.
  This matches the threat model in which an attacker keeps whichever
  inverse-folded sequence re-folds most faithfully.
- `mean` — average across samples.
- `all` — treat every sample as an independent trial.

Report which one was used; `best` is the most favourable to detection and
`all` the most conservative.

## Input format

Step 1 takes a manifest of watermarked structures:

```csv
structure_id,pdb_path,message
s0001,/data/wm/s0001.pdb,10110011010110100110100110101101
```

`message` is the 32-bit string embedded when the structure was produced; Step 3
needs it to count matching bits, so record it at fold time.

## Running

```bash
pip install -r experiments_revision/inverse_fold_refold/requirements.txt
git clone https://github.com/dauparas/ProteinMPNN

python experiments_revision/inverse_fold_refold/step1_inverse_fold.py \
    --manifest watermarked_structures.csv --mpnn_dir ./ProteinMPNN --num_seq 8

# Steps 2 and 3 need the FoldMark-Protenix root on the path for the
# protenix / decoder arms
python experiments_revision/inverse_fold_refold/step2_clean_refold.py \
    --sequences experiments_revision/inverse_fold_refold/outputs/step1/inverse_folded_sequences.csv \
    --folders protenix,esmfold

python experiments_revision/inverse_fold_refold/step3_detect_and_roc.py \
    --manifest experiments_revision/inverse_fold_refold/outputs/step2/refolded_manifest.csv \
    --target_fpr 1e-5 --aggregate best
```

Adding the AlphaFold3 arm:

```bash
python .../step2_clean_refold.py --sequences ... --folders protenix,esmfold,af3 \
    --af3_command "python run_alphafold.py --sequence {sequence} --name {name} --output_dir {out_dir}"
```

Smoke test: `--limit 2` on Step 1, `--limit 4` on Steps 2 and 3.

## Outputs

| File | Contents |
|------|----------|
| `outputs/step1/inverse_folded_sequences.csv` | One row per (structure, MPNN sample) |
| `outputs/step2/refolded_manifest.csv` | One row per (structure, sample, folder): path, TM to the watermarked backbone, `ok`, error text |
| `outputs/step3/detection_scores.csv` | Per structure/sample/folder: matching bits, bit accuracy, recovered bit string, TM |
| `outputs/step3/roc_points.csv` | Full staircase: (folder, tau, FPR, TPR) |
| `outputs/step3/tpr_at_fpr.csv` | TPR at the target FPR with n, threshold, mean bit accuracy and mean TM per condition |
| `outputs/step3/inverse_refold_roc.png/.pdf` | ROC by condition |

Failures are recorded rather than dropped: Steps 1 and 2 write `ok=False` plus
the exception text, Step 3 skips those rows, and the n actually used per
condition appears in `tpr_at_fpr.csv`. Check those n values before quoting a
TPR — a curve built from few structures has coarse, step-limited resolution.

## Figure provenance

Step 3 is the canonical source for this figure. Every plotted TPR is computed
from `outputs/step3/detection_scores.csv`; no curve is parameterised by an
assumed per-bit accuracy, and the operating-point table is written alongside the
figure so each number traces back to the structures behind it.
