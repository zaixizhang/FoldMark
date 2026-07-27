#!/usr/bin/env python3
"""
32-bit user codebook: extended BCH construction + max–min selection
====================================================================

Two-stage construction of the user signature codebook.

**Stage 1 — candidate pool.**  A binary BCH code of length 31 with designed
error-correcting capability t = 2 has parameters [31, 21, 5].  Adding an overall
parity bit extends it to **[32, 21, 6]**: 2^21 = 2,097,152 codewords, minimum
Hamming distance 6.  The pool is therefore already large enough for 10^6 users,
and *every* pair of candidates is at distance >= 6 by construction — the minimum
distance is a property of the code, not of the selection.

The generator polynomial is derived here rather than hard-coded:
g(x) = lcm(m_1(x), m_3(x)) over GF(2^5) with primitive polynomial x^5 + x^2 + 1,
where m_i is the minimal polynomial of alpha^i.  `verify_minimum_distance()`
confirms d_min = 6 by exhaustive enumeration of all 2^21 codewords.

**Stage 2 — max–min (farthest-first) selection.**  From the pool, users are
selected one at a time, each time taking the candidate whose nearest-neighbour
distance to the already-selected set is largest.  Ties are broken by the larger
mean distance to the selected set, then by a fixed seeded order.

Encoding is linear and non-systematic, c = m·G, so

    encode(m1) XOR encode(m2) = encode(m1 XOR m2)

which is what makes both the selection and the nearest-neighbour statistics
tractable at this scale: the distance between two candidates depends only on the
XOR of their message indices, so a single precomputed weight table of length
2^21 answers every distance query by lookup.
"""
from __future__ import annotations

import numpy as np

# GF(2^5) with primitive polynomial x^5 + x^2 + 1.
GF_M = 5
GF_N = 2 ** GF_M - 1          # 31
PRIM_POLY = 0b100101          # x^5 + x^2 + 1

CODE_N = 32                   # after extension
CODE_K = 21
DESIGNED_D = 6

POOL_SIZE = 2 ** CODE_K       # 2,097,152


# ── GF(2^5) arithmetic ────────────────────────────────────────────────────────

def build_gf_tables() -> tuple[np.ndarray, np.ndarray]:
    """exp[i] = alpha^i (i = 0..30), log[v] = discrete log of v."""
    exp = np.zeros(GF_N, dtype=np.int64)
    log = np.zeros(2 ** GF_M, dtype=np.int64)
    x = 1
    for i in range(GF_N):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & (1 << GF_M):
            x ^= PRIM_POLY
    return exp, log


def poly_mul_gf2(a: list[int], b: list[int]) -> list[int]:
    """Multiply two GF(2) polynomials given as coefficient lists (LSB first)."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                out[i + j] ^= bj
    return out


def minimal_polynomial(i: int, exp: np.ndarray, log: np.ndarray) -> list[int]:
    """
    Minimal polynomial over GF(2) of alpha^i, from its cyclotomic coset
    {i, 2i, 4i, ...} mod 31.  Returned LSB-first.
    """
    coset, x = [], i % GF_N
    while x not in coset:
        coset.append(x)
        x = (2 * x) % GF_N

    # Product over the coset of (z - alpha^j), computed in GF(2^5).
    poly = [1]
    for j in coset:
        root = int(exp[j])
        new = [0] * (len(poly) + 1)
        for d, c in enumerate(poly):
            if c:
                new[d + 1] ^= c                       # c * z^(d+1)
                new[d] ^= gf_mul(c, root, exp, log)   # c * alpha^j
        poly = new
    # Coefficients must now be 0/1 (the product over a full coset is binary).
    assert all(c in (0, 1) for c in poly), f"non-binary minimal polynomial: {poly}"
    return poly


def gf_mul(a: int, b: int, exp: np.ndarray, log: np.ndarray) -> int:
    if a == 0 or b == 0:
        return 0
    return int(exp[(log[a] + log[b]) % GF_N])


def generator_polynomial() -> list[int]:
    """g(x) = lcm(m_1, m_3) for the t = 2 BCH code of length 31 → degree 10."""
    exp, log = build_gf_tables()
    m1 = minimal_polynomial(1, exp, log)
    m3 = minimal_polynomial(3, exp, log)
    g = poly_mul_gf2(m1, m3)
    while len(g) > 1 and g[-1] == 0:
        g.pop()
    assert len(g) - 1 == CODE_N - 1 - CODE_K, (
        f"generator degree {len(g)-1}, expected {CODE_N - 1 - CODE_K}"
    )
    return g


# ── code construction ─────────────────────────────────────────────────────────

def generator_rows() -> np.ndarray:
    """
    The K basis codewords as uint32, bit i of the word = coefficient of x^i.

    Row j is x^j * g(x), extended with an overall parity bit in position 31.
    """
    g = generator_polynomial()
    rows = np.zeros(CODE_K, dtype=np.uint32)
    for j in range(CODE_K):
        word = 0
        for d, c in enumerate(g):
            if c:
                word |= 1 << (j + d)
        # Overall parity bit → lifts the minimum distance from 5 to 6.
        if bin(word).count("1") % 2 == 1:
            word |= 1 << (CODE_N - 1)
        rows[j] = word
    return rows


def popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorised population count for uint32 arrays."""
    a = arr.astype(np.uint32)
    a = a - ((a >> np.uint32(1)) & np.uint32(0x55555555))
    a = (a & np.uint32(0x33333333)) + ((a >> np.uint32(2)) & np.uint32(0x33333333))
    a = (a + (a >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((a * np.uint32(0x01010101)) >> np.uint32(24)).astype(np.int64)


def build_pool() -> tuple[np.ndarray, np.ndarray]:
    """
    Encode every one of the 2^21 messages.

    Returns (codewords, weights) as arrays of length 2^21, indexed by the
    integer message.  Because the encoding is linear, `weights[m1 ^ m2]` is the
    Hamming distance between candidates m1 and m2 — the identity every other
    routine in this module relies on.
    """
    rows = generator_rows()
    idx = np.arange(POOL_SIZE, dtype=np.uint32)
    codewords = np.zeros(POOL_SIZE, dtype=np.uint32)
    for j in range(CODE_K):
        bit = (idx >> np.uint32(j)) & np.uint32(1)
        codewords ^= np.where(bit == 1, rows[j], np.uint32(0))
    return codewords, popcount(codewords)


def verify_minimum_distance(weights: np.ndarray) -> int:
    """
    Minimum distance of a linear code = minimum weight of a non-zero codeword.
    Exhaustive over all 2^21 codewords, so this is a proof rather than a claim.
    """
    return int(weights[1:].min())


def weight_distribution(weights: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(weights, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


# ── stage 2: max–min selection ────────────────────────────────────────────────

def select_maxmin(
    weights: np.ndarray,
    n_users: int,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Farthest-first selection of `n_users` message indices from the pool.

    d[c] holds each candidate's nearest-neighbour distance to the selected set.
    At each step the candidate maximising d is added and d is updated by
    `d = minimum(d, weights[candidates ^ new])`, using the linearity identity.

    Early exit: the pool's minimum distance is `d_min`, so once max(d) has
    fallen to d_min no remaining choice can do better than d_min — every
    surviving candidate is equivalent under the max–min objective. The
    remainder is then filled in a seeded random order, which is equivalent to
    continuing the greedy loop but avoids ~10^6 redundant full-pool scans.
    Ties before that point are broken by the larger running mean distance, then
    by the same seeded order.
    """
    rng = np.random.default_rng(seed)
    pool_n = weights.size
    d_min = int(weights[1:].min())

    # Compact dtypes throughout: every distance fits in an int8, and the loop is
    # bandwidth-bound on full-pool passes, so this is what makes 10^6 tractable.
    w8 = weights.astype(np.int8)
    order = rng.permutation(pool_n).astype(np.int64)   # reproducible tie-break
    rank = np.empty(pool_n, dtype=np.int64)
    rank[order] = np.arange(pool_n)

    selected = np.empty(n_users, dtype=np.int64)
    sum_d = np.zeros(pool_n, dtype=np.int32)           # for the mean-distance tie-break
    idx = np.arange(pool_n, dtype=np.uint32)
    xor_buf = np.empty(pool_n, dtype=np.uint32)        # preallocated, reused each step

    first = int(order[0])
    selected[0] = first
    np.bitwise_xor(idx, np.uint32(first), out=xor_buf)
    d = w8[xor_buf]
    sum_d += d
    d[first] = -1

    n_done = 1
    while n_done < n_users:
        best = int(d.max())
        if best <= d_min:
            break

        cand = np.flatnonzero(d == best)
        if cand.size > 1:
            best_sum = sum_d[cand].max()
            cand = cand[sum_d[cand] == best_sum]
            if cand.size > 1:
                cand = cand[[int(np.argmin(rank[cand]))]]
        pick = int(cand[0])

        selected[n_done] = pick
        n_done += 1

        np.bitwise_xor(idx, np.uint32(pick), out=xor_buf)
        dist_new = w8[xor_buf]
        np.minimum(d, dist_new, out=d)
        sum_d += dist_new
        d[pick] = -1

        if verbose and n_done % 2000 == 0:
            print(f"    selected {n_done:>9,} / {n_users:,}   current max NN distance = {best}")

    if n_done < n_users:
        if verbose:
            print(f"    max NN distance reached the pool minimum ({d_min}) after "
                  f"{n_done:,} users; filling the remaining "
                  f"{n_users - n_done:,} in seeded order")
        remaining = order[~np.isin(order, selected[:n_done])]
        selected[n_done:] = remaining[: n_users - n_done]

    return selected


# ── nearest-neighbour statistics ──────────────────────────────────────────────

def nearest_neighbour_distances(
    selected: np.ndarray,
    weights: np.ndarray,
    max_radius: int = 16,
) -> np.ndarray:
    """
    Exact nearest-neighbour Hamming distance for every selected codeword.

    Uses the linearity identity: the distance between selected messages m and m'
    is `weights[m ^ m']`, so for each low-weight pool element v (in increasing
    weight) we test membership of `m ^ v` in the selected set. The first hit
    gives that codeword's nearest-neighbour distance. Selected-set membership is
    a 2^21 boolean lookup, so each test is O(1).
    """
    pool_n = weights.size
    member = np.zeros(pool_n, dtype=bool)
    member[selected] = True

    nn = np.full(selected.size, -1, dtype=np.int64)
    todo = np.arange(selected.size)

    for radius in range(1, max_radius + 1):
        if todo.size == 0:
            break
        vs = np.flatnonzero(weights == radius).astype(np.uint32)
        if vs.size == 0:
            continue
        cur = selected[todo].astype(np.uint32)
        hit = np.zeros(todo.size, dtype=bool)
        # Chunk over v to bound peak memory at ~|todo| * chunk entries.
        chunk = max(1, int(4_000_000 // max(1, cur.size)))
        for s in range(0, vs.size, chunk):
            block = vs[s : s + chunk]
            hit |= member[(cur[:, None] ^ block[None, :])].any(axis=1)
            if hit.all():
                break
        nn[todo[hit]] = radius
        todo = todo[~hit]

    return nn
