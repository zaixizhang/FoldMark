#!/usr/bin/env python3
"""
Step 4 — End-to-end encode → fold → detect → attribute
=======================================================

Steps 1–3 are coding theory and statistics. This step is the actual experiment:
take a user's 32-bit signature from the codebook, **embed it** by folding a
protein with FoldMark-Protenix, **decode it** back out of the resulting
structure, and **attribute** the structure to a user through the codebook.

    user id ──► codebook lookup ──► 32-bit message
                                        │
                                        ▼
        sequence ──► FoldMark-Protenix (watermark=message) ──► structure
                                        │
                                        ▼
                     FoldMark decoder ──► 32 recovered bits
                                        │
                                        ▼
              nearest codeword + radius/margin guards ──► attributed user

What it reports per sample: the recovered bits, the per-sample bit accuracy,
the Hamming distance to the true user's codeword, the distance to the nearest
and second-nearest codewords, and the attribution outcome
(`correct` / `wrong` / `abstain`).

The aggregate **per-bit error rate** printed at the end is the empirical number
that Step 2's simulation should be parameterised with — run this first on real
structures, then feed the measured rate into
`step2_tracing_simulation.py --error_rates <measured>` to extrapolate attribution
behaviour across the full user population without folding 10^6 proteins.

Embedding a *specific* message
------------------------------
FoldMark samples a random code during training
(`batch['watermark'] = torch.randint(0, 2, (watermark_emb,))`), and the released
inference path exposes `configs.watermark` as a boolean.  Embedding a chosen
user signature therefore needs the message tensor to be set explicitly.
`set_watermark_message()` below tries the known injection points in order and
raises a clear error naming them if none is present, rather than silently
falling back to a random code — a silent fallback would make every number in
this script meaningless.

If your checkpoint exposes a different hook, wire it in `set_watermark_message()`
only; nothing else in this file needs to change.

Input
-----
CSV of folding jobs:

    sample_id,user_id,sequence
    s0001,17,MISLIAALAVDRVIGMENAMPWNLP...
    s0002,4211,GESLFKGPRDYNPISSTICHLTNES...

`user_id` indexes into the Step 1 codebook (row number, 0-based).

Output
------
outputs/step4/
    roundtrip_results.csv     one row per sample
    roundtrip_summary.json    aggregate bit error rate and attribution rates
    structures/               the watermarked predictions

Run from the FoldMark-Protenix root, so `configs/` and `runner/` import:
    python experiments_revision/user_tracing_codebook/step4_encode_detect_roundtrip.py \\
        --jobs jobs.csv \\
        --codebook experiments_revision/user_tracing_codebook/outputs/step1/codebook.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402

DEFAULT_CODEBOOK = (
    Path(__file__).resolve().parent / "outputs" / "step1" / "codebook.npy"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step4"


# ── codebook helpers ──────────────────────────────────────────────────────────

def load_codebook_words(path: Path) -> np.ndarray:
    """Step 1 message indices → 32-bit codewords as uint32."""
    messages = np.load(path).astype(np.uint32)
    pool, _ = cb.build_pool()
    return pool[messages]


def word_to_bits(word: np.uint32, n_bits: int = cb.CODE_N) -> np.ndarray:
    return np.array([(int(word) >> i) & 1 for i in range(n_bits)], dtype=np.int8)


def bits_to_word(bits: np.ndarray) -> np.uint32:
    out = 0
    for i, b in enumerate(np.asarray(bits)[: cb.CODE_N]):
        if int(b):
            out |= 1 << i
    return np.uint32(out)


def attribute(query: np.uint32, words: np.ndarray, radius: int, margin: int):
    """(nearest index, d1, d2, attributed?) under the radius/margin guards."""
    dist = cb.popcount(words ^ query)
    i1 = int(np.argmin(dist))
    d1 = int(dist[i1])
    saved = dist[i1]
    dist[i1] = np.iinfo(np.int64).max
    d2 = int(dist.min())
    dist[i1] = saved
    return i1, d1, d2, (d1 <= radius and (d2 - d1) >= margin)


# ── FoldMark embedding / decoding ─────────────────────────────────────────────

def set_watermark_message(configs, runner, bits: np.ndarray) -> None:
    """
    Install a specific 32-bit message as the watermark to embed.

    Tries the known injection points in order:
      1. `configs.watermark_message`      — if the config schema exposes it
      2. `runner.model.watermark_message` — attribute on the flow model
      3. `runner.model.set_watermark(...)` — explicit setter, if defined

    Raises if none is available. Do NOT replace this with a silent fallback to a
    random code: every number this script produces assumes the embedded message
    is the user's signature.
    """
    import torch

    msg = torch.tensor(np.asarray(bits, dtype=np.float32))

    if hasattr(configs, "watermark_message"):
        configs.watermark_message = msg
        return

    model = getattr(runner, "model", None)
    if model is not None:
        if hasattr(model, "set_watermark") and callable(model.set_watermark):
            model.set_watermark(msg)
            return
        if hasattr(model, "watermark_message"):
            model.watermark_message = msg
            return

    raise RuntimeError(
        "Could not install a specific watermark message. None of the expected "
        "hooks were found:\n"
        "  configs.watermark_message\n"
        "  runner.model.set_watermark(tensor)\n"
        "  runner.model.watermark_message\n"
        "Wire your checkpoint's hook into set_watermark_message() in this file. "
        "Refusing to continue, because folding with a random code instead of the "
        "user's signature would silently invalidate every result here."
    )


def fold_with_message(sequence: str, name: str, out_dir: Path, bits: np.ndarray,
                      checkpoint: str, seed: int = 101) -> Path:
    """Fold `sequence` with FoldMark, embedding `bits`. Returns the CIF path."""
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import (
        InferenceRunner,
        download_infercence_cache,
        infer_predict,
    )
    from runner.msa_search import update_infer_json

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(
        [{"sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
          "name": name}], indent=2))

    arg_str = (f"--seeds {seed} --dump_dir {out_dir} "
               f"--input_json_path {json_path} --model.N_cycle 10 "
               f"--sample_diffusion.N_sample 1 --sample_diffusion.N_step 200")
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str=arg_str,
                            fill_required_with_null=True)
    configs.load_checkpoint_path = checkpoint
    download_infercence_cache()
    configs.input_json_path = update_infer_json(str(json_path), str(out_dir.parent), True)
    configs.watermark = True
    configs.saved_path = str(out_dir)

    runner = InferenceRunner(configs)
    set_watermark_message(configs, runner, bits)
    infer_predict(runner, configs)

    cifs = sorted((out_dir / "predictions").glob("**/*.cif"))
    if not cifs:
        raise FileNotFoundError(f"no CIF produced under {out_dir}")
    return cifs[0]


def detect_bits(cif_path: Path, checkpoint: str, n_bits: int = cb.CODE_N) -> np.ndarray:
    """Decode the watermark from a structure (mirrors runner.inference.infer_detect)."""
    import torch
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, get_inference_dataloader

    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, arg_str="", fill_required_with_null=True)
    configs.load_checkpoint_path = checkpoint
    configs.input_json_path = str(cif_path)

    runner = InferenceRunner(configs)
    for batch in get_inference_dataloader(configs=configs):
        prediction, _label = runner.detect(batch)
        code = torch.clamp(prediction["watermark"], min=-10, max=10)
        return (torch.sigmoid(code) > 0.5).long().cpu().numpy().ravel()[:n_bits]
    raise RuntimeError(f"decoder produced no output for {cif_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 4: encode a user code, fold, detect, and attribute"
    )
    ap.add_argument("--jobs",       type=Path, required=True,
                    help="CSV: sample_id,user_id,sequence")
    ap.add_argument("--codebook",   type=Path, default=DEFAULT_CODEBOOK)
    ap.add_argument("--out_dir",    type=Path, default=OUT_DIR)
    ap.add_argument("--checkpoint", type=str,  default="./checkpoint.pt")
    ap.add_argument("--radius",     type=int,  default=2)
    ap.add_argument("--margin",     type=int,  default=1)
    ap.add_argument("--seed",       type=int,  default=101)
    ap.add_argument("--limit",      type=int,  default=None)
    args = ap.parse_args()

    if not args.codebook.is_file():
        sys.exit(f"[error] codebook not found: {args.codebook}\n"
                 "        Run step1_build_codebook.py first.")
    if not args.jobs.is_file():
        sys.exit(f"[error] jobs CSV not found: {args.jobs}")

    jobs = pd.read_csv(args.jobs)
    for col in ("sample_id", "user_id", "sequence"):
        if col not in jobs.columns:
            sys.exit(f"[error] jobs CSV is missing the {col!r} column")
    if args.limit:
        jobs = jobs.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    struct_dir = args.out_dir / "structures"
    struct_dir.mkdir(exist_ok=True)

    words = load_codebook_words(args.codebook)
    n_users = words.size
    print(f"[step4] Codebook: {n_users:,} users, {cb.CODE_N}-bit signatures")
    print(f"[step4] {len(jobs)} folding jobs")
    print(f"[step4] Rule: attribute if d1 <= {args.radius} and (d2 - d1) >= {args.margin}")

    rows: list[dict] = []
    for i, job in enumerate(jobs.itertuples(index=False), start=1):
        sid  = str(job.sample_id)
        uid  = int(job.user_id)
        if not 0 <= uid < n_users:
            print(f"  [{i}] {sid}: user_id {uid} outside the codebook — skipped")
            continue

        true_word = words[uid]
        true_bits = word_to_bits(true_word)

        try:
            cif = fold_with_message(str(job.sequence), sid, struct_dir / sid,
                                    true_bits, args.checkpoint, seed=args.seed)
            rec_bits = detect_bits(cif, args.checkpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(jobs)}] {sid}: FAILED ({type(exc).__name__}: {exc})")
            rows.append({"sample_id": sid, "user_id": uid, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        k = min(rec_bits.size, true_bits.size)
        n_match = int((rec_bits[:k] == true_bits[:k]).sum())
        query = bits_to_word(rec_bits)
        i1, d1, d2, attributed = attribute(query, words, args.radius, args.margin)

        if not attributed:
            outcome = "abstain"
        elif i1 == uid:
            outcome = "correct"
        else:
            outcome = "wrong"

        rows.append({
            "sample_id": sid, "user_id": uid, "ok": True, "error": "",
            "n_bits": k, "n_matching_bits": n_match,
            "bit_accuracy": n_match / k,
            "hamming_to_true_user": k - n_match,
            "nearest_user": i1, "d1": d1, "d2": d2,
            "attributed": attributed, "outcome": outcome,
            "recovered_bits": "".join(str(int(b)) for b in rec_bits[:k]),
            "cif_path": str(cif),
        })
        print(f"  [{i}/{len(jobs)}] {sid:<12s} user={uid:<8d} "
              f"bits={n_match}/{k}  d1={d1} d2={d2}  {outcome}")

    if not rows:
        sys.exit("[error] no jobs completed")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "roundtrip_results.csv", index=False)

    ok = df[df["ok"] == True]  # noqa: E712
    summary: dict = {
        "n_jobs": int(len(df)),
        "n_succeeded": int(len(ok)),
        "n_users": int(n_users),
        "radius": args.radius,
        "margin": args.margin,
    }
    if len(ok):
        per_bit_error = 1.0 - float(ok["bit_accuracy"].mean())
        counts = ok["outcome"].value_counts().to_dict()
        summary.update({
            "mean_bit_accuracy": float(ok["bit_accuracy"].mean()),
            "per_bit_error_rate": per_bit_error,
            "mean_hamming_to_true_user": float(ok["hamming_to_true_user"].mean()),
            "correct_rate": counts.get("correct", 0) / len(ok),
            "wrong_attribution_rate": counts.get("wrong", 0) / len(ok),
            "abstain_rate": counts.get("abstain", 0) / len(ok),
            "outcome_counts": {k: int(v) for k, v in counts.items()},
        })
    (args.out_dir / "roundtrip_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Round-trip summary ({len(ok)}/{len(df)} succeeded) ===")
    if len(ok):
        print(f"  Mean bit accuracy        : {summary['mean_bit_accuracy']:.4f}")
        print(f"  Per-bit error rate       : {summary['per_bit_error_rate']:.4f}")
        print(f"  Correct attribution      : {summary['correct_rate']:.1%}")
        print(f"  Wrong attribution        : {summary['wrong_attribution_rate']:.2%}")
        print(f"  Abstained                : {summary['abstain_rate']:.1%}")
        print(f"\n  Feed the measured error rate into Step 2:\n"
              f"    python step2_tracing_simulation.py --codebook {args.codebook} "
              f"--error_rates {summary['per_bit_error_rate']:.4f}\n")
    print(f"  Written to {args.out_dir}\n")


if __name__ == "__main__":
    main()
