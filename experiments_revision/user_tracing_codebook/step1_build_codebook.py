#!/usr/bin/env python3
"""
Step 1 — Build the million-user codebook and report its geometry
=================================================================

Constructs the 32-bit user signature codebook in the two stages described in
codebook.py:

  1. Candidate pool: extended BCH [32, 21, 6] — 2^21 = 2,097,152 codewords with
     minimum Hamming distance 6.  The generator polynomial is derived from the
     cyclotomic cosets rather than hard-coded, and the minimum distance is
     **verified exhaustively** over all 2^21 codewords rather than asserted.

  2. User selection: deterministic max–min (farthest-first) over the pool, ties
     broken by larger mean distance to the selected set and then by a fixed
     seeded order.

Everything here is deterministic given `--seed`, and no model or data is
involved — this step is pure coding theory and runs on CPU.

Output
------
outputs/step1/
    codebook.npy              selected message indices (uint32, n_users)
    codebook_bits.npy         the codewords themselves as a uint8 (n_users, 32)
    pool_weight_distribution.csv
    nn_distance_distribution.csv
    codebook_summary.json     parameters, verified d_min, NN statistics

Run:
    python experiments_revision/user_tracing_codebook/step1_build_codebook.py \\
        --n_users 1000000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import codebook as cb  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs" / "step1"


def unpack_codewords(codewords: np.ndarray, n_bits: int = cb.CODE_N) -> np.ndarray:
    """uint32 codewords → (n, n_bits) uint8 bit matrix, bit i in column i."""
    out = np.zeros((codewords.size, n_bits), dtype=np.uint8)
    for i in range(n_bits):
        out[:, i] = ((codewords >> np.uint32(i)) & np.uint32(1)).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 1: build the million-user 32-bit codebook"
    )
    ap.add_argument("--n_users", type=int, default=1_000_000)
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip_verify", action="store_true",
                    help="skip the exhaustive minimum-distance check")
    ap.add_argument("--compare_random", action="store_true",
                    help="also score a uniformly random selection of the same "
                         "size, to show what the max-min procedure actually buys")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: candidate pool ───────────────────────────────────────────────
    print(f"[step1] Extended BCH [{cb.CODE_N}, {cb.CODE_K}, {cb.DESIGNED_D}] "
          f"over GF(2^{cb.GF_M}), primitive polynomial {bin(cb.PRIM_POLY)}")
    g = cb.generator_polynomial()
    print(f"[step1] Generator polynomial degree {len(g)-1}, coefficients (LSB first): {g}")

    t0 = time.time()
    codewords, weights = cb.build_pool()
    print(f"[step1] Candidate pool: {codewords.size:,} codewords "
          f"({time.time()-t0:.1f}s)")

    if args.n_users > codewords.size:
        sys.exit(f"[error] requested {args.n_users:,} users but the pool holds "
                 f"only {codewords.size:,}")

    d_min = None
    if not args.skip_verify:
        d_min = cb.verify_minimum_distance(weights)
        print(f"[step1] Verified minimum distance (exhaustive over all "
              f"{codewords.size:,} codewords): d_min = {d_min}")
        if d_min != cb.DESIGNED_D:
            sys.exit(f"[error] expected d_min = {cb.DESIGNED_D}, got {d_min}")

    wd = cb.weight_distribution(weights)
    with (args.out_dir / "pool_weight_distribution.csv").open("w") as f:
        f.write("weight,count\n")
        for w, c in sorted(wd.items()):
            f.write(f"{w},{c}\n")

    # ── Stage 2: max–min selection ───────────────────────────────────────────
    print(f"\n[step1] Max–min selection of {args.n_users:,} users (seed={args.seed}) …")
    t0 = time.time()
    selected = cb.select_maxmin(weights, args.n_users, seed=args.seed, verbose=True)
    sel_seconds = time.time() - t0
    print(f"[step1] Selection complete in {sel_seconds:.1f}s")

    if np.unique(selected).size != selected.size:
        sys.exit("[error] selection produced duplicate codewords")

    # ── Nearest-neighbour geometry ───────────────────────────────────────────
    print("[step1] Computing exact nearest-neighbour distances …")
    t0 = time.time()
    nn = cb.nearest_neighbour_distances(selected, weights)
    print(f"[step1] Nearest-neighbour distances computed in {time.time()-t0:.1f}s")

    if (nn < 0).any():
        sys.exit("[error] some nearest-neighbour distances were not resolved; "
                 "increase max_radius in nearest_neighbour_distances()")

    vals, counts = np.unique(nn, return_counts=True)
    with (args.out_dir / "nn_distance_distribution.csv").open("w") as f:
        f.write("nn_hamming_distance,count,fraction\n")
        for v, c in zip(vals, counts):
            f.write(f"{int(v)},{int(c)},{c/nn.size:.8f}\n")

    # ── optional: what does max–min actually buy at this scale? ──────────────
    random_summary = None
    if args.compare_random:
        rng = np.random.default_rng(args.seed)
        rnd = rng.choice(codewords.size, size=args.n_users, replace=False)
        rnd_nn = cb.nearest_neighbour_distances(rnd, weights)
        rvals, rcounts = np.unique(rnd_nn, return_counts=True)
        random_summary = {
            "nn_min": int(rnd_nn.min()),
            "nn_mean": float(rnd_nn.mean()),
            "nn_distribution": {int(v): int(c) for v, c in zip(rvals, rcounts)},
        }
        print(f"[step1] Uniformly random selection of the same size: "
              f"NN min = {rnd_nn.min()}, mean = {rnd_nn.mean():.4f}")
        if np.isclose(rnd_nn.mean(), nn.mean()):
            print("[step1] NOTE: at this n the max–min selection gives the same "
                  "nearest-neighbour distribution as random selection — the pool "
                  "is saturated and the minimum distance of the code is the "
                  "binding constraint.")

    sel_codewords = codewords[selected]
    np.save(args.out_dir / "codebook.npy", selected.astype(np.uint32))
    np.save(args.out_dir / "codebook_bits.npy", unpack_codewords(sel_codewords))

    summary = {
        "code": f"extended BCH [{cb.CODE_N}, {cb.CODE_K}, {cb.DESIGNED_D}]",
        "primitive_polynomial": bin(cb.PRIM_POLY),
        "generator_polynomial_lsb_first": g,
        "pool_size": int(codewords.size),
        "verified_pool_min_distance": d_min,
        "n_users": int(args.n_users),
        "seed": int(args.seed),
        "selection_seconds": round(sel_seconds, 1),
        "nn_min": int(nn.min()),
        "nn_max": int(nn.max()),
        "nn_mean": float(nn.mean()),
        "nn_median": float(np.median(nn)),
        "nn_distribution": {int(v): int(c) for v, c in zip(vals, counts)},
    }
    if random_summary is not None:
        summary["random_selection_baseline"] = random_summary
    (args.out_dir / "codebook_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\n=== Codebook summary ===\n"
        f"  Code                   : extended BCH [{cb.CODE_N}, {cb.CODE_K}, {cb.DESIGNED_D}]\n"
        f"  Candidate pool         : {codewords.size:,}\n"
        f"  Verified pool d_min    : {d_min}\n"
        f"  Users selected         : {args.n_users:,}\n"
        f"  NN distance min        : {nn.min()}\n"
        f"  NN distance mean       : {nn.mean():.4f}\n"
        f"  NN distribution        : "
        f"{ {int(v): int(c) for v, c in zip(vals, counts)} }\n"
        f"\n  Written to {args.out_dir}\n"
        f"  Pass to Step 2:  --codebook {args.out_dir/'codebook.npy'}\n"
    )


if __name__ == "__main__":
    main()
