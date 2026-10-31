"""Edit matched, already circulated sequences. Positions are 1-based in CSV output."""
import argparse
import csv
import random
from pathlib import Path

from common import METHODS, OPS, IDENTITIES, digest, output_file
from editors import AA, blosum, mpnn, EsmEditor

FIELDS = ["sample_id", "split", "method", "label", "sequence"]
EXTRA = ["run_id", "seed", "operation", "identity", "actual_identity", "positions",
         "edited_sequence", "row_id"]


def run(args):
    with args.manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    if not rows or set(FIELDS) - set(columns) or set(EXTRA) & set(columns):
        raise ValueError(f"Empty or malformed manifest. Required columns: {FIELDS}")
    keys, lengths, splits = set(), {}, {}
    for row in rows:
        sid, split, method, label, seq = [row[k] for k in FIELDS]
        key = (sid, method, label)
        if not sid or not seq or set(seq) - set(AA) or method not in METHODS or label not in ("0", "1"):
            raise ValueError(f"Invalid manifest row: {key}")
        if split not in ("calibration", "test") or splits.setdefault(sid, split) != split:
            raise ValueError(f"Invalid or overlapping split: {sid}")
        if key in keys or lengths.setdefault(sid, len(seq)) != len(seq):
            raise ValueError(f"Duplicate row or unaligned pair: {key}")
        keys.add(key)
        for name in ("backbone", "reference"):
            if row.get(name):
                row[name] = str((args.manifest.parent / row[name]).resolve())
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.operators)) != len(args.operators):
        raise ValueError("Use distinct seeds and operators")
    esm = None
    with output_file(args.output) as f:
        writer = csv.DictWriter(f, [*columns, *EXTRA])
        writer.writeheader()
        for row in rows:
            seq = row["sequence"]
            for run_id, seed in enumerate(args.seeds, start=1):
                for op in args.operators:
                    # Omit method and class from the seed. Paired arms get identical sites.
                    edit_seed = int(digest(seed, row["sample_id"], op)[:8], 16) % (2**31-1) + 1
                    order = list(range(len(seq)))
                    random.Random(edit_seed).shuffle(order)
                    for identity in IDENTITIES:
                        count = ((100 - identity) * len(seq) + 50) // 100
                        sites = order[:count]
                        if not sites:
                            edited = seq
                        elif op == "blosum62":
                            edited = blosum(seq, sites, edit_seed)
                        elif op == "esm2":
                            if not args.esm_checkpoint:
                                raise ValueError("Provide --esm-checkpoint for ESM2")
                            if esm is None:
                                esm = EsmEditor(args.esm_checkpoint, args.device)
                            edited = esm.edit(seq, sites)
                        else:
                            if not args.mpnn_dir or not row.get("backbone"):
                                raise ValueError("ProteinMPNN requires --mpnn-dir and backbone")
                            edited = mpnn(seq, sites, edit_seed, row["backbone"],
                                          row.get("chain") or "A", args.mpnn_dir,
                                          args.temperature, args.mpnn_model)
                        actual = [i for i, (a, b) in enumerate(zip(seq, edited)) if a != b]
                        if len(edited) != len(seq) or set(edited) - set(AA) or set(actual) != set(sites):
                            raise ValueError(f"Edit budget mismatch for {row['sample_id']}, {op}, {identity}")
                        row_id = digest(row["sample_id"], row["method"], row["label"], seed,
                                        op, identity, seq, edited)
                        writer.writerow(dict(row, run_id=run_id, seed=seed, operation=op,
                                             identity=identity, actual_identity=100*(1-count/len(seq)),
                                             positions=" ".join(str(i+1) for i in sites),
                                             edited_sequence=edited, row_id=row_id))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--operators", nargs="+", choices=OPS, default=OPS)
    p.add_argument("--esm-checkpoint", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mpnn-dir", type=Path)
    p.add_argument("--mpnn-model", default="v_48_020")
    p.add_argument("--temperature", type=float, default=0.1)
    args = p.parse_args()
    if args.temperature <= 0:
        p.error("--temperature must be positive")
    run(args)
