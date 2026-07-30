#!/usr/bin/env python3
"""
Multi-bit wrapper around the Chen et al. sequence-level watermark
==================================================================

Chen et al. (`protein_watermark`, https://github.com/poseidonchan/ProteinWatermark)
implement a **zero-bit** watermark: `WatermarkDetector.detect()` returns a
detection score and a p-value answering "is this sequence watermarked?", but it
carries no payload.  FoldMark, by contrast, embeds a **32-bit message** and
reports *bit recovery accuracy*.

To compare the two on a matched footing we lift Chen et al. to a multi-bit
scheme using the standard **message-as-key (key-partition)** construction:

  · The m-bit message is split into `n_chunks` chunks of `bits_per_chunk` bits.
  · The sequence is split into `n_chunks` contiguous segments.
  · Segment j is sampled using a private key derived from
        HMAC-SHA256(master_key, j || chunk_value_j)
    so the *identity of the key* encodes the chunk's value.
  · At detection, segment j is scored under all `2 ** bits_per_chunk` candidate
    keys; the argmax is the recovered chunk value.

With the defaults (32 chunks × 1 bit) this yields the same 32-bit payload as
FoldMark's `watermark_emb: 32`, so bit-recovery accuracy is directly comparable.

Note this construction is *favourable* to Chen et al. — it is the strongest
obvious way to give a zero-bit scheme a payload.  It also makes explicit why
short sequences are hard: at L = 100 with 32 chunks each segment is only ~3
residues, which is little evidence per bit.

Dependencies:
    pip install protein-watermark
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np

try:
    from protein_watermark import (
        DeltaGumbel_Reweight,
        WatermarkDetector,
        WatermarkLogitsProcessor,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "protein_watermark is required for the Chen et al. baseline.\n"
        "    pip install protein-watermark"
    ) from exc


DEFAULT_MASTER_KEY = b"FoldMark-vs-Chen-comparison"
DEFAULT_N_CHUNKS = 32       # matches FoldMark's 32-bit payload
DEFAULT_BITS_PER_CHUNK = 1
DEFAULT_CONTEXT_CODE_LENGTH = 5   # Chen et al. default


def derive_key(master_key: bytes, chunk_index: int, chunk_value: int) -> bytes:
    """Private key for one (chunk index, chunk value) pair."""
    msg = f"{chunk_index}:{chunk_value}".encode()
    return hmac.new(master_key, msg, hashlib.sha256).digest()


def bits_to_chunks(bits: np.ndarray, bits_per_chunk: int) -> list[int]:
    """[1,0,1,...] -> per-chunk integer values (MSB first within a chunk)."""
    bits = np.asarray(bits).astype(int).ravel()
    if bits.size % bits_per_chunk:
        raise ValueError(
            f"message length {bits.size} is not divisible by bits_per_chunk={bits_per_chunk}"
        )
    return [
        int("".join(str(b) for b in bits[i : i + bits_per_chunk]), 2)
        for i in range(0, bits.size, bits_per_chunk)
    ]


def chunks_to_bits(chunks: list[int], bits_per_chunk: int) -> np.ndarray:
    out: list[int] = []
    for value in chunks:
        out.extend(int(b) for b in format(value, f"0{bits_per_chunk}b"))
    return np.array(out, dtype=int)


def segment_bounds(length: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split [0, length) into n_chunks near-equal contiguous segments."""
    edges = np.linspace(0, length, n_chunks + 1).astype(int)
    return [(int(edges[j]), int(edges[j + 1])) for j in range(n_chunks)]


@dataclass
class ChenMultiBitConfig:
    master_key: bytes = DEFAULT_MASTER_KEY
    n_chunks: int = DEFAULT_N_CHUNKS
    bits_per_chunk: int = DEFAULT_BITS_PER_CHUNK
    context_code_length: int = DEFAULT_CONTEXT_CODE_LENGTH
    vocab_size: int = 32

    @property
    def n_bits(self) -> int:
        return self.n_chunks * self.bits_per_chunk

    @property
    def n_candidates(self) -> int:
        return 2 ** self.bits_per_chunk


class ChenMultiBitWatermarker:
    """
    Multi-bit encoder/decoder built on Chen et al.'s logits processor + detector.

    Encoding is *online*: `processor_for_position` returns the
    WatermarkLogitsProcessor that should reweight the logits at the given
    generation step, given the message being embedded.
    """

    def __init__(self, cfg: ChenMultiBitConfig | None = None) -> None:
        self.cfg = cfg or ChenMultiBitConfig()

    # ── encoding ──────────────────────────────────────────────────────────────

    def build_processors(self, message_bits: np.ndarray) -> list[WatermarkLogitsProcessor]:
        """One processor per chunk, keyed by that chunk's value."""
        chunk_values = bits_to_chunks(message_bits, self.cfg.bits_per_chunk)
        if len(chunk_values) != self.cfg.n_chunks:
            raise ValueError(
                f"expected {self.cfg.n_chunks} chunks, got {len(chunk_values)}"
            )
        return [
            WatermarkLogitsProcessor(
                derive_key(self.cfg.master_key, j, value),
                DeltaGumbel_Reweight(),
                context_code_length=self.cfg.context_code_length,
            )
            for j, value in enumerate(chunk_values)
        ]

    def chunk_index_for_position(self, pos: int, total_length: int) -> int:
        for j, (lo, hi) in enumerate(segment_bounds(total_length, self.cfg.n_chunks)):
            if lo <= pos < hi:
                return j
        return self.cfg.n_chunks - 1

    def apply(
        self,
        processors: list[WatermarkLogitsProcessor],
        context_ids: np.ndarray,
        logits: np.ndarray,
        pos: int,
        total_length: int,
    ) -> np.ndarray:
        """
        Reweight `logits` (shape [batch, vocab]) for generation step `pos`.
        `context_ids` is the token history, shape [batch, t].
        """
        j = self.chunk_index_for_position(pos, total_length)
        return processors[j](
            mode="normal",
            context=context_ids,
            logits=logits,
        )

    # ── decoding ──────────────────────────────────────────────────────────────

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Recover the message from a tokenized sequence, shape [1, L] or [L].

        For each segment, score it under every candidate key and take the argmax.
        """
        token_ids = np.atleast_2d(np.asarray(token_ids))
        length = token_ids.shape[1]
        bounds = segment_bounds(length, self.cfg.n_chunks)

        recovered: list[int] = []
        for j, (lo, hi) in enumerate(bounds):
            segment = token_ids[:, lo:hi]
            if segment.shape[1] < 2:
                # Segment too short to score — emit a chunk value of 0.
                # This is a genuine failure mode at short lengths, not an error.
                recovered.append(0)
                continue

            best_value, best_score = 0, -np.inf
            for value in range(self.cfg.n_candidates):
                detector = WatermarkDetector(
                    derive_key(self.cfg.master_key, j, value),
                    DeltaGumbel_Reweight(),
                    context_code_length=self.cfg.context_code_length,
                    vocab_size=self.cfg.vocab_size,
                )
                try:
                    score, _p = detector.detect(segment)
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_value, best_score = value, score
            recovered.append(best_value)

        return chunks_to_bits(recovered, self.cfg.bits_per_chunk)

    def bit_accuracy(self, true_bits: np.ndarray, token_ids: np.ndarray) -> float:
        pred = self.decode(token_ids)
        true = np.asarray(true_bits).astype(int).ravel()
        n = min(pred.size, true.size)
        return float((pred[:n] == true[:n]).mean())


def random_message(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random bit message — matches FoldMark's torch.randint(0, 2, ...)."""
    return rng.integers(0, 2, size=n_bits).astype(int)
