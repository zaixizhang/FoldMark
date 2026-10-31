# Extended Data Fig. 5 | Matched sequence circulation

This reference implementation starts at the point where each method has produced a sequence to share. It implements the three sequence-editing conditions in panels a–c, exports clean AlphaFold 3 jobs, and calculates detection and structural fidelity from per-protein measurements. The algorithms and defaults below specify this implementation. A numerical comparison against the submitted figure requires the authors' archived inputs, detector settings and measured outputs.

Run the examples from this directory with Python 3.10+. Install `requirements.txt` in the analysis environment. For editing, install [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) with its vanilla weights, PyTorch, and [fair-esm](https://github.com/facebookresearch/esm) with a local ESM2 checkpoint. Use the native environments for AF3 and each detector. Record dependency versions, upstream commits, checkpoint hashes, seeds and input file checksums with a production run. The numeric seeds below are examples.

## 1. Supply the paired inputs

Use one CSV row per source protein, method and class. A source ID identifies the same aligned, equal-length monomer in all arms. Include matched non-watermarked pipeline controls. Keep separate source protein IDs for calibration and testing. The strict analysis expects matching positive and negative test observations for every test protein. Calibration needs at least 1,000 distinct negative controls per condition for the default empirical operating point.

| Column | Value |
|---|---|
| `sample_id,split` | Source protein ID and `calibration` or `test` |
| `method,label` | `chen`, `synthid_seq`, `synthid_struct`, or `foldmark`; 1 for watermark present, 0 for control |
| `sequence` | Unedited circulated sequence, canonical uppercase amino acids |
| `backbone,chain` | Single-chain design input for ProteinMPNN (PDB, chain A by default). Coordinates and residue identities must correspond to the circulated sequence; use its clean predicted backbone when necessary. |
| `reference,reference_chain` | Fidelity reference for this arm, with the chain specified (A by default). Paths are relative to the manifest or absolute. Record which reference structure was selected. |
| `detector_context` | Optional generation-time detector context. For the supplied FoldMark scorer use the registered 32-bit code; assign the corresponding code to matched controls too. |

Chen and SynthIDBio-sequence use sequence-level watermarking. For FoldMark and SynthIDBio-structure the circulated input is the sequence inverse-folded from the watermarked structure. Precomputed inputs preserve the actual sequences and paired controls without requiring regeneration during the editing step. Supply real generation-time settings and keys via local files.

## 2. Edit the circulated sequences

```bash
python step1_edit.py data/inputs.csv work/edits.csv \
  --seeds 11 22 33 44 55 --mpnn-dir /path/to/ProteinMPNN \
  --esm-checkpoint /path/to/esm2_t33_650M_UR50D.pt --device cuda
```

For a CPU-only run of panel a, pass `--operators blosum62` and omit the model options. For all three operators a seed derived from (run seed, source protein ID, operator) shuffles positions once. The arms and their matched controls share the ordered sites, with nested prefixes at 100%, 95%, 90% and 80% identity. The mutation count is the nearest attainable integer to the target, with half counts rounded up. Outputs record the one-based positions in update order, realized identity, both sequences, and a stable row ID. Generation fails on a mismatch between intended and observed edits.

| Operator | Rule in this reference implementation |
|---|---|
| BLOSUM62 | At each selected position choose the highest-scoring non-self canonical residue; sample uniformly among ties with the paired seed. For residues without a positive non-self score this uses the best available score. |
| ProteinMPNN | Run vanilla ProteinMPNN on the supplied backbone and chain. Fix every position outside the mask, omit the original residue at each selected site, draw one sequence, and check the resulting Hamming distance. Defaults: v_48_020, temperature 0.1. |
| ESM2 | Visit selected positions in paired order. Mask one residue in the current sequence, choose the highest conditional-logit canonical alternative, and update the context before the next position. This optimizes a local conditional objective subject to the edit budget. |

## 3. Refold with clean AF3

```bash
python step2_af3.py work/edits.csv work/mapping.csv --jobs work/af3_inputs --results work/af3_results
# In the original AF3 environment, with its standard unwatermarked weights and databases:
python /path/to/alphafold3/run_alphafold.py --input_dir=work/af3_inputs \
  --output_dir=work/af3_results --model_dir=/path/to/weights --db_dir=/path/to/databases
python step2_af3.py work/edits.csv work/mapping.csv --jobs work/af3_inputs \
  --results work/af3_results --require-results
```

Use a fresh job directory per dataset. Jobs are shared only when edited sequence and AF3 seed agree. The mapping records the expected top-ranked CIF for each row. Set paths according to your AF3 environment or container mounts.

## 4. Score samples

Run each method in its own environment or import measured native scores. Example invocation:

```bash
python step3_score.py work/mapping.csv work/chen.csv --method chen --config /path/to/chen.json
```

- **Chen.** Install the upstream `protein-watermark` and `tokenizers` packages. Supply JSON fields `tokenizer` (ProGen2 tokenizer JSON), `key_file` (exact raw generation-key bytes), `context_length` and `vocab_size`. Detection uses the native whole-sequence score with history reset for each sequence.
- **SynthIDBio-sequence.** Install the [upstream SynthIDBio-sequence environment](https://github.com/google-deepmind/synthidbio). Supply `processor` with the generation-time `keys`, `ngram_len`, `context_history_size`, `temperature`, `top_k`, `skip_first_ngram_calls`, and `num_leaves`. The scorer averages native token g-values over the edited sequence in full precision.
- **FoldMark.** Use the inference environment specified in [inverse_fold_refold](../inverse_fold_refold/), making its FoldMark-Protenix modules importable via PYTHONPATH. Supply `checkpoint`. The example presence statistic is the number of recovered bits matching the registered code in the clean AF3 structure. Calibration below is empirical. To use another presence detector, provide its scores or factory.
- **SynthIDBio-structure.** Supply the authorized native detector and weights for scoring a *clean refolded CIF*. For this environment use a local Python factory specified as `{"factory":"your_module:load_detector","settings":{...}}`. The factory takes the settings dictionary and returns a callable `score(cif_path, detector_context) -> float`. Access to a watermark generator alone does not supply a standalone detection score. Alternatively import the native detector's recorded scores. The same factory interface is supported for all four methods.

For panel d the scoring stage TM-aligns the top-ranked clean AF3 chain to the designated reference, normalized by reference length, for positive test rows at target 90% identity. When importing original measurements, use the reference and normalization from that experiment. Every score CSV has columns `row_id,score,scTM`, with higher detection scores indicating stronger evidence for a watermark. Other rows can leave scTM empty. The plotting stage requires exactly one score for every mapped row. Failed or missing model outputs must be resolved before analysis; they are never scored as zero.

## 5. Regenerate the panels from per-sample measurements

```bash
python step4_figure.py work/mapping.csv --scores work/chen.csv work/synthid_seq.csv \
  work/synthid_struct.csv work/foldmark.csv --out work/figure
```

For a fast replot, start at this command with the archived mapping and score files. The default design is 1,000 test proteins and five paired runs. Each run, method, operator and target identity has its own threshold estimated using held-out negatives. A score must strictly exceed the (floor(0.001 × n) + 1)-th largest calibration score; ties are handled conservatively. The observed calibration and test FPRs are written alongside TPR. The nominal 0.1% target is not rounded into an observed rate.

`test_decisions.csv` links each test call to a row ID and threshold. `per_run.csv` reports denominators, TPR, observed FPR, and the panel-d mean scTM per run. `extended_data_fig5.pdf` shows the mean and sample SD across runs for each condition. Significance annotations require the original comparison specification and measured data and are not generated here.
