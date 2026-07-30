# Watermark persistence under structural post-processing

Does the FoldMark watermark survive the refinement steps a structure would
routinely be put through after prediction — and does it still survive when the
protein is in a **bound state**, with a ligand or an antibody partner present?

This directory evaluates four conditions on watermarked complexes:

| Method | What it does | Expected structural impact |
|--------|--------------|----------------------------|
| `none` | Unprocessed baseline | — |
| `amber` | Amber relaxation (OpenFold/AlphaFold relaxer, default settings) | Local, mostly side-chain and clash relief |
| `rosetta_fastrelax` | Rosetta FastRelax, default script | Local backbone + side-chain refinement |
| `rosetta_ngk` | Next-generation KIC loop modeling (Stein & Kortemme 2013) | Aggressive local backbone remodeling of a loop |

Both bound-state contexts from Figure 4 of the main paper are covered:
**protein–ligand** and **protein–antibody** complexes. `--iterations N` repeats
an operator N times to cover single vs. iterative application.

```
Step 1 ── Apply each operator to each watermarked complex
              ↓  post-processed PDBs + postprocessed_manifest.csv
Step 2 ── Decode the watermark from every processed structure
              ↓  detection_scores.csv  (bits matching the embedded message)
Step 3 ── ROC with analytic null → TPR @ FPR, and the two-panel figure
```

## Detection statistic and the null

The decoder emits K bits; the test statistic is **k**, the number matching the
embedded message. Under H0 (unwatermarked structure, or the wrong key) the bits
are fair coins, so the false-positive rate is available in closed form:

```
FPR(tau) = P[X >= tau],  X ~ Binomial(K, 0.5)
```

This is what makes an operating point as strict as **FPR = 1e-5** quotable
without needing 10^5 negative structures, and it is the null validated by the
per-bit calibration analysis (decoded bits on unwatermarked structures sit at
~0.5 per position and are near-uncorrelated).

**TPR is empirical** — the fraction of watermarked structures whose k reaches
the threshold. Every plotted point is measured; only the FPR axis is analytic.

One consequence worth checking before quoting a number: a K-bit code cannot
express an FPR below `2^-K`. FoldMark uses **K = 32** throughout
(`watermark_emb: 32`), for which the smallest attainable analytic FPR is
2.3e-10, so FPR = 1e-5 is comfortably reachable. Step 2 still reads K from the
decoder rather than assuming it, and Step 3 warns if the requested FPR is below
what K supports, so a shorter payload degrades loudly rather than silently.

## Bound-state handling

The operators run on the complex as given, so the refinement sees the binding
partner. Two caveats that affect interpretation:

- **Amber** — the bundled OpenFold relaxer models protein atoms only.
  Non-protein records are held out of the minimisation and re-attached
  unchanged, so the ligand acts as fixed context rather than being relaxed
  itself. Relaxing the ligand too needs an explicit small-molecule
  parameterisation (OpenFF/GAFF); that path is stubbed as
  `--amber_backend openmm_complex` and is not required to reproduce the figure.
- **Rosetta** — ligands need params. Pass `--params lig.params` or rely on
  `-load_PDB_components` for standard components. PyRosetta is initialised once
  per process with `-ignore_unrecognized_res false` so a missing ligand
  definition fails loudly instead of being silently dropped.

Residues are not renumbered or reordered by any operator, because Step 2's
decoder relies on residue correspondence with the input structure.

## Input format

Step 1 takes a manifest of already-watermarked structures:

```csv
structure_id,dataset,pdb_path,message
7abc_A,ligand,/data/wm/7abc_wm.pdb,10110011010110100110100110101101
5xyz_H,antibody,/data/wm/5xyz_wm.pdb,01001101011010011010011010110100
```

- `dataset` — `ligand` or `antibody`
- `message` — the ground-truth bit string embedded when the structure was
  produced. Step 2 needs it to count matching bits, so record it at fold time.

## Running

```bash
pip install -r experiments_revision/robustness_postprocessing/requirements.txt
# PyRosetta separately (academic licence): https://www.pyrosetta.org/downloads

python experiments_revision/robustness_postprocessing/step1_postprocess_structures.py \
    --manifest my_watermarked_structures.csv \
    --methods none,amber,rosetta_fastrelax,rosetta_ngk

# Step 2 must run from the FoldMark-Protenix root so configs/ and runner/ import
python experiments_revision/robustness_postprocessing/step2_detect_watermark.py \
    --manifest experiments_revision/robustness_postprocessing/outputs/step1/postprocessed_manifest.csv

python experiments_revision/robustness_postprocessing/step3_roc_and_figure.py \
    --scores_csv experiments_revision/robustness_postprocessing/outputs/step2/detection_scores.csv \
    --target_fpr 1e-5
```

Smoke test on a couple of structures, skipping the tools you have not installed:

```bash
python .../step1_postprocess_structures.py --manifest m.csv --methods none,amber --limit 2
python .../step2_detect_watermark.py --limit 8
```

Iterative application:

```bash
python .../step1_postprocess_structures.py --manifest m.csv --iterations 5
```

## Outputs

| File | Contents |
|------|----------|
| `outputs/step1/postprocessed_manifest.csv` | One row per (structure, method): output path, wall-clock seconds, `ok`, exception text on failure |
| `outputs/step2/detection_scores.csv` | Per structure and method: K, matching bits, bit accuracy |
| `outputs/step3/roc_points.csv` | Full staircase: (dataset, method, tau, FPR, TPR) |
| `outputs/step3/tpr_at_fpr.csv` | TPR at the target FPR, with n, threshold, achieved FPR, mean ± s.d. bit accuracy |
| `outputs/step3/relaxation_roc.png/.pdf` | Two-panel ROC figure: protein–ligand and protein–antibody |

Post-processing failures are recorded rather than dropped silently: Step 1
writes `ok=False` plus the exception, Step 2 skips those rows, and Step 3
reports the n actually used per curve in `tpr_at_fpr.csv`. Check those n values
before quoting a TPR — a curve built from few structures has a coarse,
step-limited TPR resolution.

## Figure provenance

Step 3 is the canonical source for this figure. Every plotted TPR is computed
from `outputs/step2/detection_scores.csv`; no curve is parameterised by an
assumed per-bit accuracy, and the operating-point table is written alongside the
figure so each number can be traced to the structures behind it.

## References

1. Case, D. A. et al. The Amber biomolecular simulation programs. *J. Comput. Chem.* **26**, 1668–1688 (2005).
2. Das, R. & Baker, D. Macromolecular modeling with Rosetta. *Annu. Rev. Biochem.* **77**, 363–382 (2008).
3. Stein, A. & Kortemme, T. Improvements to robotics-inspired conformational sampling in Rosetta. *PLoS One* **8**, e63090 (2013).
