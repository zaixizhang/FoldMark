# Supplementary experiments

Code for the supplementary analyses accompanying the manuscript. Each
subdirectory is self-contained and produces the figure and statistics for one
analysis.

## Contents

| Directory | Experiment | Manuscript item |
|-----------|------------|-----------------|
| [`chen_et_al_comparison/`](./chen_et_al_comparison) | FoldMark vs. a protein-specific sequence-level watermarking baseline (Chen et al.) | Bit recovery accuracy and scTM vs. sequence length |
| [`robustness_postprocessing/`](./robustness_postprocessing) | Watermark persistence under Amber relaxation, Rosetta FastRelax and NGK loop modeling, in protein–ligand and protein–antibody bound states | Detection ROC per post-processing method |
| [`user_tracing_codebook/`](./user_tracing_codebook) | Million-user codebook construction (extended BCH + max–min selection), attribution under decoding errors, and calibration of the decoder null | Nearest-neighbour distance statistics, attribution rates, null-bias diagnostics |
| [`inverse_fold_refold/`](./inverse_fold_refold) | Watermark survival through ProteinMPNN inverse folding followed by clean re-folding with Protenix, ESMFold or AlphaFold3 | Detection ROC per re-folding model, with TM-score to the watermarked backbone |

## Conventions

Each subdirectory follows the same layout as `tutorials/`:

- numbered `stepN_*.py` scripts that run in order, each writing to
  `outputs/stepN/` and passing an explicit file path to the next;
- a `README.md` stating the setup, the metrics, the exact commands, and the
  justification for any methodological choice the comparison rests on;
- a `requirements.txt` listing only what that experiment adds on top of the
  base FoldMark environment.

Statistical tests report the test used, n, the effect (difference in medians),
the raw p-value and a multiple-comparison-corrected p-value. Figure scripts
read their numbers from the upstream step's CSV rather than hard-coding summary
statistics, so a figure can always be traced back to the data that produced it.
