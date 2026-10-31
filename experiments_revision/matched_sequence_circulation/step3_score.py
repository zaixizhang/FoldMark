"""Score a single detector arm per invocation, in that model's own environment."""
import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

from common import METHODS, output_file


def detector(method, cfg):
    if "factory" in cfg:
        module, name = cfg["factory"].split(":")
        # The factory returns score(sequence_or_cif, public_context). Do not pass labels.
        return getattr(importlib.import_module(module), name)(cfg.get("settings", {}))
    if method == "chen":
        import numpy as np
        from tokenizers import Tokenizer
        from protein_watermark import DeltaGumbel_Reweight, WatermarkDetector
        tokenizer = Tokenizer.from_file(cfg["tokenizer"])
        model = WatermarkDetector(Path(cfg["key_file"]).read_bytes(), DeltaGumbel_Reweight(),
                                  cfg["context_length"], cfg["vocab_size"])

        def score(seq, context):
            ids = tokenizer.encode(seq, add_special_tokens=False).ids
            if len(ids) != len(seq):
                raise ValueError("Chen requires the generation-time residue tokenizer")
            model.reset_history()
            value, _ = model.detect(np.asarray([ids], dtype=np.int64))
            return float(value)
        return score
    if method == "synthid_seq":
        import torch
        import jax.numpy as jnp
        from synthid_text.detector_mean import mean_score
        from synthid_text.logits_processing import SynthIDLogitsProcessor
        from editors import AA
        params = dict(cfg["processor"])
        device = torch.device(params.pop("device", "cpu"))
        model = SynthIDLogitsProcessor(device=device, apply_top_k=False, **params)

        def score(seq, context):
            tokens = torch.tensor([[AA.index(aa) for aa in seq]], device=device)
            g = model.compute_g_values(tokens).detach().cpu().numpy()
            return float(mean_score(jnp.asarray(g), jnp.ones(g.shape[:2]))[0])
        return score
    if method == "foldmark":
        # Reuse the sibling workflow's decoder on the clean prediction. Context is
        # the registered code for this test, assigned to positives and controls alike.
        sibling = Path(__file__).resolve().parents[1] / "inverse_fold_refold"
        sys.path.insert(0, str(sibling))
        from step3_detect_and_roc import decode_bits, load_runner
        runner, settings = load_runner(cfg["checkpoint"])

        def score(cif, context):
            if len(context) != 32 or set(context) - {"0", "1"}:
                raise ValueError("FoldMark reference detector requires a registered 32-bit code")
            bits = decode_bits(runner, settings, Path(cif), n_bits=32)
            if len(bits) != 32:
                raise ValueError("Incomplete FoldMark decoder output")
            return sum(int(b) == int(c) for b, c in zip(bits, context))
        return score
    raise ValueError("Supply a factory for the installed structure detector, or import its measured scores")


def sc_tm(reference, ref_chain, mobile, sequence):
    import numpy as np
    from Bio.PDB import MMCIFParser, PDBParser, Polypeptide
    from tmtools import tm_align

    def trace(path, chain):
        parser = MMCIFParser if Path(path).suffix.lower() in (".cif", ".mmcif") else PDBParser
        residues = [r for r in parser(QUIET=True).get_structure("p", path)[0][chain]
                    if r.id[0] == " " and "CA" in r]
        seq = "".join(Polypeptide.protein_letters_3to1[r.resname] for r in residues)
        coords = np.asarray([r["CA"].coord for r in residues], dtype=np.float64)
        return seq, coords

    ref, x = trace(reference, ref_chain)
    pred, y = trace(mobile, "A")
    if pred != sequence or len(ref) != len(sequence):
        raise ValueError("Check reference and AF3 residue coverage/chain selection")
    return float(tm_align(x, y, ref, pred).tm_norm_chain1)


def validate_foldmark_codes(mapping):
    """A registered query code must also be used for the matched negative."""
    codes = {}
    with mapping.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != "foldmark":
                continue
            code = row.get("detector_context", "")
            if len(code) != 32 or set(code) - {"0", "1"}:
                raise ValueError("Missing or invalid registered FoldMark code")
            pair = (row["sample_id"], row["run_id"])
            if codes.setdefault(pair, code) != code:
                raise ValueError(f"Positive and control must query the same registered code: {pair}")


def main(args):
    cfg = json.loads(args.config.read_text())
    if args.method == "foldmark" and "factory" not in cfg:
        validate_foldmark_codes(args.mapping)
    score = detector(args.method, cfg)
    seen = set()
    with args.mapping.open(newline="") as src, output_file(args.output) as out:
        writer = csv.DictWriter(out, fieldnames=["row_id", "score", "scTM"])
        writer.writeheader()
        for row in csv.DictReader(src):
            if row["method"] != args.method:
                continue
            rid = row["row_id"]
            if rid in seen:
                raise ValueError(f"Duplicate row {rid}")
            seen.add(rid)
            seq, cif = row["edited_sequence"], row["refolded_cif"]
            seq_detector = args.method in ("chen", "synthid_seq")
            if not seq_detector and not Path(cif).is_file():
                raise FileNotFoundError(cif)
            value = float(score(seq if seq_detector else cif, row.get("detector_context", "")))
            tm = ""
            if (row["split"], row["label"], row["identity"]) == ("test", "1", "90"):
                tm = sc_tm(row["reference"], row.get("reference_chain") or "A", cif, seq)
                if not 0 <= tm <= 1:
                    raise ValueError(f"Invalid scTM for {rid}")
            if not math.isfinite(value):
                raise ValueError(f"Invalid score for {rid}")
            writer.writerow(dict(row_id=rid, score=value, scTM=tm))
        if not seen:
            raise ValueError("No rows for selected method")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mapping", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--config", type=Path, required=True)
    main(p.parse_args())
