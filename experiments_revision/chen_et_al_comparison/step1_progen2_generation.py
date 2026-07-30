#!/usr/bin/env python3
"""
Step 1 — Matched ProGen2 sequence generation for both arms
===========================================================

Following the setting of Chen et al., protein sequences are generated with
**ProGen2** under identical conditions for both arms — same model, same
sampling temperature, same lengths, same number of sequences per length, and
the **same per-sequence RNG seed**, so that sequence *i* in one arm is the
paired counterpart of sequence *i* in the other.  That pairing is what makes
the two-sided Wilcoxon **signed-rank** test in Step 3 valid.

Two arms:

  chen      — sequences are watermarked *at the sequence level* during
              sampling, by reweighting the ProGen2 logits at each step with
              Chen et al.'s WatermarkLogitsProcessor (multi-bit wrapper in
              chen_watermark.py).  These are folded with the ORIGINAL
              structure predictor in Step 2.

  foldmark  — sequences are generated normally, with NO sequence-level
              watermark.  The watermark is instead applied at the structure
              prediction stage in Step 2, by folding with FoldMark-Protenix.

Both arms embed/expect the same 32-bit payload, matching FoldMark's
`watermark_emb: 32`.

Output : outputs/step1/
           sequences_chen.fasta
           sequences_foldmark.fasta
           manifest.csv     ← per-sequence: arm, length, seed, message bits

Run:
    python experiments_revision/chen_et_al_comparison/step1_progen2_generation.py \\
        --lengths 100,200,300,400 --n_per_length 1000 --temperature 1.0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chen_watermark import (  # noqa: E402
    DEFAULT_N_CHUNKS,
    ChenMultiBitConfig,
    ChenMultiBitWatermarker,
    random_message,
)

# ProGen2 checkpoint. The Salesforce release is also mirrored on the Hub; any
# causal-LM checkpoint with a residue-level tokenizer works here.
DEFAULT_MODEL = "hugohrban/progen2-base"

DEFAULT_LENGTHS = [100, 200, 300, 400]
DEFAULT_N_PER_LENGTH = 1000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SEED = 0

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step1"

# ProGen2 emits sequences prefixed with a control token; "1" is the N->C tag.
BOS_TOKEN = "1"


def load_progen2(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[step1] Loading ProGen2: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model = model.eval().to(device)
    print(f"[step1] ProGen2 ready on {device}; vocab={model.config.vocab_size}")
    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    device: str,
    length: int,
    temperature: float,
    seed: int,
    watermarker: ChenMultiBitWatermarker | None = None,
    message_bits: np.ndarray | None = None,
) -> tuple[str, np.ndarray]:
    """
    Autoregressively sample one sequence of `length` residues.

    If `watermarker` is given, Chen et al.'s reweighting is applied to the
    logits at every step, embedding `message_bits`.  Otherwise sampling is
    plain temperature sampling.

    Returns (amino-acid string, token id array).
    """
    import torch

    torch.manual_seed(seed)
    processors = (
        watermarker.build_processors(message_bits) if watermarker is not None else None
    )

    input_ids = tokenizer(BOS_TOKEN, return_tensors="pt").input_ids.to(device)
    generated: list[int] = []

    with torch.no_grad():
        past = None
        cur = input_ids
        for pos in range(length):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float() / temperature

            if processors is not None:
                context = np.atleast_2d(
                    np.array(generated, dtype=np.int64)
                    if generated
                    else input_ids.cpu().numpy().ravel()
                )
                reweighted = watermarker.apply(
                    processors,
                    context_ids=context,
                    logits=logits.cpu().numpy(),
                    pos=pos,
                    total_length=length,
                )
                logits = torch.from_numpy(np.asarray(reweighted)).to(device).float()

            probs = torch.softmax(logits, dim=-1)
            # DeltaGumbel reweighting makes the distribution a point mass;
            # multinomial handles both that and the plain case.
            next_id = torch.multinomial(probs, num_samples=1)
            generated.append(int(next_id.item()))
            cur = next_id

    token_ids = np.array(generated, dtype=np.int64)
    seq = tokenizer.decode(generated)
    seq = "".join(ch for ch in seq.upper() if ch.isalpha())
    return seq, token_ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 1: matched ProGen2 generation for the Chen et al. comparison"
    )
    ap.add_argument("--model_id",     type=str, default=DEFAULT_MODEL)
    ap.add_argument("--lengths",      type=str, default=",".join(map(str, DEFAULT_LENGTHS)))
    ap.add_argument("--n_per_length", type=int, default=DEFAULT_N_PER_LENGTH)
    ap.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    ap.add_argument("--n_chunks",     type=int, default=DEFAULT_N_CHUNKS,
                    help="payload chunks (32 x 1 bit = 32-bit message)")
    ap.add_argument("--bits_per_chunk", type=int, default=1)
    ap.add_argument("--context_code_length", type=int, default=5)
    ap.add_argument("--out_dir",      type=Path, default=OUT_DIR)
    ap.add_argument("--device",       type=str, default="auto")
    args = ap.parse_args()

    import torch
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_progen2(args.model_id, device)

    cfg = ChenMultiBitConfig(
        n_chunks=args.n_chunks,
        bits_per_chunk=args.bits_per_chunk,
        context_code_length=args.context_code_length,
        vocab_size=int(model.config.vocab_size),
    )
    watermarker = ChenMultiBitWatermarker(cfg)
    print(f"[step1] Payload: {cfg.n_bits} bits "
          f"({cfg.n_chunks} chunks x {cfg.bits_per_chunk} bit)")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    fa_chen     = (args.out_dir / "sequences_chen.fasta").open("w")
    fa_foldmark = (args.out_dir / "sequences_foldmark.fasta").open("w")

    for length in lengths:
        print(f"\n[step1] Length {length} aa — {args.n_per_length} sequences per arm")
        for i in range(args.n_per_length):
            # One seed and one message per PAIR, shared by both arms.
            pair_seed = int(rng.integers(0, 2 ** 31 - 1))
            message   = random_message(cfg.n_bits, rng)
            msg_str   = "".join(map(str, message))
            sid       = f"L{length}_{i:04d}"

            # chen arm — watermark embedded during sampling
            seq_chen, ids_chen = generate_one(
                model, tokenizer, device, length, args.temperature, pair_seed,
                watermarker=watermarker, message_bits=message,
            )
            fa_chen.write(f">{sid}|arm=chen|len={length}|seed={pair_seed}|msg={msg_str}\n{seq_chen}\n")

            # foldmark arm — no sequence-level watermark
            seq_fm, ids_fm = generate_one(
                model, tokenizer, device, length, args.temperature, pair_seed,
            )
            fa_foldmark.write(f">{sid}|arm=foldmark|len={length}|seed={pair_seed}|msg={msg_str}\n{seq_fm}\n")

            rows.append({
                "seq_id": sid, "length": length, "seed": pair_seed, "message": msg_str,
                "sequence_chen": seq_chen, "sequence_foldmark": seq_fm,
                "token_ids_chen": " ".join(map(str, ids_chen)),
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1:4d}/{args.n_per_length}] length={length}")

    fa_chen.close()
    fa_foldmark.close()

    manifest = args.out_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seq_id","length","seed","message","sequence_chen",
                        "sequence_foldmark","token_ids_chen"],
        )
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n[step1] Done. {len(rows)} matched pairs across {len(lengths)} lengths.\n"
        f"  FASTA (chen)     → {args.out_dir/'sequences_chen.fasta'}\n"
        f"  FASTA (foldmark) → {args.out_dir/'sequences_foldmark.fasta'}\n"
        f"  Manifest         → {manifest}\n"
        f"\n  Pass to Step 2:  --manifest {manifest}\n"
    )


if __name__ == "__main__":
    main()
