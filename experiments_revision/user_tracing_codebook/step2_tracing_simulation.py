#!/usr/bin/env python3
"""
Step 2 — Attribution under decoding errors
===========================================

Given the codebook from Step 1, this quantifies what happens to user attribution
when the decoder makes bit errors — the case the tracing claim depends on.

Attribution rule
----------------
A decoded 32-bit string y is attributed by **minimum Hamming distance** to the
codebook, subject to two guards:

    d1 = distance to the nearest codeword
    d2 = distance to the second nearest codeword

    attribute to the nearest user   if  d1 <= radius  AND  (d2 - d1) >= margin
    abstain                         otherwise

The `radius` guard bounds how far a decode may drift before it is trusted at
all; the `margin` guard refuses to attribute when two users are almost equally
close.  That second guard is what matters at 10^6 users: many signatures sit
near each other, so an unguarded nearest-neighbour rule will confidently name
the wrong user once errors accumulate. Reporting abstentions separately keeps
"attributed to the wrong user" distinct from "declined to attribute".

Outcomes reported per per-bit error rate p:

    correct    attributed to the true user
    wrong      attributed to a different user   ← the number that matters
    abstain    no attribution made

A null arm is also run: decoded bits drawn from the H0 model (fair coins by
default, or an empirical per-position bias via `--null_bits_csv`) are pushed
through the same rule, giving the **false attribution rate** for a structure
that carries no watermark.

Output
------
outputs/step2/
    attribution_by_error_rate.csv
    attribution_null.csv

Run:
    python experiments_revision/user_tracing_codebook/step2_tracing_simulation.py \\
        --codebook experiments_revision/user_tracing_codebook/outputs/step1/codebook.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402

DEFAULT_CODEBOOK = (
    Path(__file__).resolve().parent / "outputs" / "step1" / "codebook.npy"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step2"

DEFAULT_ERROR_RATES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_codebook_words(path: Path) -> np.ndarray:
    """Load Step 1 message indices and re-encode them to 32-bit codewords."""
    messages = np.load(path).astype(np.uint32)
    codewords_all, _ = cb.build_pool()
    return codewords_all[messages]


def two_nearest(query: np.uint32, words: np.ndarray) -> tuple[int, int, int]:
    """(nearest index, d1, d2) of `query` against the codebook."""
    dist = cb.popcount(words ^ query)
    i1 = int(np.argmin(dist))
    d1 = int(dist[i1])
    saved = dist[i1]
    dist[i1] = np.iinfo(np.int64).max
    d2 = int(dist.min())
    dist[i1] = saved
    return i1, d1, d2


def flip_bits(word: np.uint32, p: float, rng: np.random.Generator) -> np.uint32:
    """Independent bit flips with probability p across the 32 positions."""
    mask = 0
    for b in range(cb.CODE_N):
        if rng.random() < p:
            mask |= 1 << b
    return np.uint32(int(word) ^ mask)


def bits_to_word(bits: np.ndarray) -> np.uint32:
    out = 0
    for i, b in enumerate(bits[: cb.CODE_N]):
        if int(b):
            out |= 1 << i
    return np.uint32(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 2: attribution accuracy under decoder bit errors"
    )
    ap.add_argument("--codebook",  type=Path, default=DEFAULT_CODEBOOK)
    ap.add_argument("--out_dir",   type=Path, default=OUT_DIR)
    ap.add_argument("--error_rates", type=str,
                    default=",".join(str(p) for p in DEFAULT_ERROR_RATES))
    ap.add_argument("--radius",  type=int, default=2,
                    help="max Hamming distance to accept an attribution")
    ap.add_argument("--margin",  type=int, default=1,
                    help="required gap between nearest and second-nearest")
    ap.add_argument("--n_trials", type=int, default=2000)
    ap.add_argument("--n_null",   type=int, default=2000)
    ap.add_argument("--null_bits_csv", type=Path, default=None,
                    help="CSV with a 'bits' column of decoded strings from "
                         "unwatermarked structures; used instead of fair coins")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.codebook.is_file():
        sys.exit(f"[error] codebook not found: {args.codebook}\n"
                 "        Run step1_build_codebook.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    words = load_codebook_words(args.codebook)
    n_users = words.size
    print(f"[step2] Codebook: {n_users:,} users, {cb.CODE_N}-bit signatures")
    print(f"[step2] Rule: attribute if d1 <= {args.radius} and (d2 - d1) >= {args.margin}")

    rng = np.random.default_rng(args.seed)
    error_rates = [float(x) for x in args.error_rates.split(",") if x.strip()]

    # ── attribution under bit errors ─────────────────────────────────────────
    rows = []
    for p in error_rates:
        users = rng.integers(0, n_users, size=args.n_trials)
        correct = wrong = abstain = 0
        d1_sum = 0

        for u in users:
            y = flip_bits(words[u], p, rng)
            i1, d1, d2 = two_nearest(y, words)
            d1_sum += d1
            if d1 <= args.radius and (d2 - d1) >= args.margin:
                if i1 == int(u):
                    correct += 1
                else:
                    wrong += 1
            else:
                abstain += 1

        n = args.n_trials
        rows.append({
            "per_bit_error_rate": p,
            "n_trials": n,
            "correct_rate": correct / n,
            "wrong_attribution_rate": wrong / n,
            "abstain_rate": abstain / n,
            "mean_d1": d1_sum / n,
            "radius": args.radius,
            "margin": args.margin,
        })
        print(f"  p={p:5.2f}  correct={correct/n:6.1%}  "
              f"wrong={wrong/n:6.2%}  abstain={abstain/n:6.1%}  "
              f"mean d1={d1_sum/n:5.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "attribution_by_error_rate.csv", index=False)

    # ── null arm: unwatermarked / non-enrolled input ─────────────────────────
    print(f"\n[step2] Null arm ({args.n_null:,} draws) …")
    if args.null_bits_csv and args.null_bits_csv.is_file():
        null_df = pd.read_csv(args.null_bits_csv)
        queries = [bits_to_word(np.array([int(c) for c in str(s).strip()]))
                   for s in null_df["bits"].head(args.n_null)]
        null_source = str(args.null_bits_csv)
    else:
        queries = [np.uint32(int(rng.integers(0, 2 ** cb.CODE_N)))
                   for _ in range(args.n_null)]
        null_source = "fair-coin H0 (Bernoulli(0.5) per position)"
    print(f"[step2] Null source: {null_source}")

    false_attr = 0
    d1_list = []
    for y in queries:
        _i1, d1, d2 = two_nearest(y, words)
        d1_list.append(d1)
        if d1 <= args.radius and (d2 - d1) >= args.margin:
            false_attr += 1

    n_null = len(queries)
    null_row = {
        "null_source": null_source,
        "n_draws": n_null,
        "false_attribution_rate": false_attr / n_null if n_null else float("nan"),
        "mean_d1": float(np.mean(d1_list)) if d1_list else float("nan"),
        "min_d1": int(np.min(d1_list)) if d1_list else -1,
        "radius": args.radius,
        "margin": args.margin,
        "n_users": int(n_users),
    }
    pd.DataFrame([null_row]).to_csv(args.out_dir / "attribution_null.csv", index=False)

    print(f"  false attribution rate : {null_row['false_attribution_rate']:.3%} "
          f"({false_attr}/{n_null})")
    print(f"  mean distance to nearest codeword : {null_row['mean_d1']:.2f}")
    print(f"  min  distance to nearest codeword : {null_row['min_d1']}")

    print(f"\n[step2] Written to {args.out_dir}\n"
          f"  Pass to Step 3:  --attribution_csv "
          f"{args.out_dir/'attribution_by_error_rate.csv'}\n")


if __name__ == "__main__":
    main()
