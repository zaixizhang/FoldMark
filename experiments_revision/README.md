# Supplementary experiments

Code for the supplementary analyses accompanying the manuscript. Each
subdirectory is self-contained and produces the figure and statistics for the
corresponding Extended Data item.

## Contents

| Directory | Experiment | Manuscript item |
|-----------|------------|-----------------|
| [`chen_et_al_comparison/`](./chen_et_al_comparison) | FoldMark vs. a protein-specific sequence-level watermarking baseline (Chen et al.) | Extended Data Fig. 1 |
| [`robustness_postprocessing/`](./robustness_postprocessing) | Watermark persistence under Amber relaxation, Rosetta FastRelax and NGK loop modeling, in protein–ligand and protein–antibody bound states | Extended Data Fig. 4 |

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
