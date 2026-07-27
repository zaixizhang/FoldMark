# Chen et al. baseline comparison (Extended Data Fig. 1 / Fig. R1)

Reviewer Comment 3 asked for a **protein-specific baseline** such as Chen et al.
This directory contains the full pipeline for that comparison, under a matched
generation and evaluation protocol.

> Chen et al., *ProteinWatermark: Enhancing privacy in secured protein design* —
> https://github.com/poseidonchan/ProteinWatermark

## The comparison

Both arms start from the **same ProGen2 generation conditions** — same model,
same sampling temperature, same lengths, same number of sequences per length,
and the **same per-sequence RNG seed and message**. The two arms differ only in
*where* the watermark is applied:

| | Chen et al. (baseline) | FoldMark (ours) |
|---|---|---|
| Sequence generation | ProGen2, **watermarked at the sequence level** during sampling | ProGen2, **unwatermarked** |
| Structure prediction | Original AlphaFold3 / Protenix | AlphaFold3 / Protenix **equipped with FoldMark** |
| Watermark lives in | The amino-acid sequence | The predicted backbone |
| Recovery from | Token sequence, via the Chen detector | Structure, via the FoldMark decoder |

Because the pairing is preserved end-to-end, sequence *i* in one arm is the
matched counterpart of sequence *i* in the other, which is what licenses the
**two-sided Wilcoxon signed-rank** (paired) test in Step 3.

```
Step 1 ── ProGen2 generation, both arms, matched seeds + messages
              ↓  sequences_chen.fasta / sequences_foldmark.fasta / manifest.csv
Step 2 ── Fold each arm with its predictor; score bit recovery + scTM
              ↓  paired_scores.csv
Step 3 ── Wilcoxon signed-rank + Holm correction → statistics.csv + Fig. R1 b,c
```

## Making a zero-bit scheme multi-bit (important)

Chen et al. is a **zero-bit** watermark: its detector answers "is this sequence
watermarked?" with a score and a p-value, but carries no payload. FoldMark
embeds a **16-bit message** (`watermark_emb: 16`) and reports *bit recovery
accuracy*. Comparing the two on that metric requires lifting Chen et al. to a
multi-bit scheme.

`chen_watermark.py` does this with the standard **message-as-key
(key-partition)** construction:

- the 16-bit message is split into `n_chunks` chunks of `bits_per_chunk` bits
  (default 16 × 1, matching FoldMark's payload);
- the sequence is split into `n_chunks` contiguous segments;
- segment *j* is sampled under a private key derived from
  `HMAC-SHA256(master_key, j ‖ chunk_value_j)`, so the *choice of key* encodes
  the chunk value;
- at detection, each segment is scored under all `2 ** bits_per_chunk` candidate
  keys and the argmax is the recovered chunk.

**This is the fairest available construction and is deliberately generous to the
baseline** — it is the strongest obvious way to give a zero-bit scheme a
payload. It also makes the short-length behaviour transparent rather than
mysterious: at L = 100 with 16 chunks, each segment is only ~6 residues, which
is very little evidence per bit. Any comparison of "bit accuracy" against a
zero-bit method rests on a choice like this one, and it should be stated in the
manuscript rather than left implicit.

If a reviewer would prefer a **zero-bit** comparison instead (detection AUROC or
TPR at fixed FPR, using Chen's native score/p-value against FoldMark's decoder
confidence), that is a different and also defensible framing — but it is not
what the current response text claims, so it would need the response rewritten.

## Metrics

- **Watermark bit recovery accuracy** — fraction of the 16 message bits
  recovered. Chen: from the token sequence. FoldMark: from the predicted
  structure, via `sigmoid(prediction['watermark']) > 0.5`, exactly as in
  `runner.inference.infer_detect` and `models/flow_module.py`.
- **scTM** — self-consistency TM-score of the predicted backbone: ProteinMPNN
  (8 designs, T = 0.1) → ESMFold → max TM-score against the backbone, computed
  with `tmtools.tm_align`.

## Running

```bash
# Dependencies (on top of the FoldMark-Protenix Space requirements)
pip install -r experiments_revision/chen_et_al_comparison/requirements.txt
git clone https://github.com/dauparas/ProteinMPNN

# Step 2 must run from the FoldMark-Protenix root so that configs/ and runner/
# are importable:
#   git clone https://huggingface.co/spaces/Zaixi/FoldMark foldmark_space

python experiments_revision/chen_et_al_comparison/step1_progen2_generation.py \
    --lengths 100,200,300,400 --n_per_length 1000 --temperature 1.0

python experiments_revision/chen_et_al_comparison/step2_fold_and_score.py \
    --manifest experiments_revision/chen_et_al_comparison/outputs/step1/manifest.csv \
    --mpnn_dir ./ProteinMPNN

python experiments_revision/chen_et_al_comparison/step3_statistics_and_figure.py \
    --paired_csv experiments_revision/chen_et_al_comparison/outputs/step2/paired_scores.csv
```

Smoke test first — Step 2 folds 2 × N structures and is by far the expensive
step (4 lengths × 1000 sequences × 2 arms = 8000 structure predictions plus the
scTM self-consistency loop):

```bash
python .../step2_fold_and_score.py --limit 4 --mpnn_dir ./ProteinMPNN
python .../step2_fold_and_score.py --limit 50 --skip_sctm   # recovery only, fast
```

## Outputs

| File | Contents |
|------|----------|
| `outputs/step1/manifest.csv` | Matched pairs: seed, 16-bit message, both sequences, Chen token ids |
| `outputs/step2/per_sequence_scores.csv` | One row per (sequence, arm): bit accuracy, scTM |
| `outputs/step2/paired_scores.csv` | One row per pair, both arms side by side |
| `outputs/step3/statistics.csv` | Per length and metric: n, means, s.d., medians, W, p, Holm-corrected p, stars |
| `outputs/step3/watermark_comparison_chen_vs_foldmark.png/.pdf` | Fig. R1 panels b, c |

Every value plotted by Step 3 is read from the Step 2 CSV — no summary
statistic is hard-coded in the figure script.

## Note on the figure in the current revision draft

`FoldMark_revision/watermark_comparison.py` (outside this repo) generates the
same two-panel figure from **hard-coded mock numbers** — its own docstring says
"Mock comparison" and the significance stars there are labelled decorative. That
script must not be the source of Extended Data Fig. 1. Step 3 here is its
replacement: same house style, but every number derived from the Step 2 CSVs.
