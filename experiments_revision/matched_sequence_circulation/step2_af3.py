"""Prepare native AlphaFold 3 monomer jobs and map their top-ranked outputs to rows."""
import argparse
import csv
import json
from pathlib import Path

from common import digest, output_file


def main(args):
    seen = set()
    with args.edits.open(newline="") as src, output_file(args.mapping) as out:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(out, fieldnames=[*reader.fieldnames, "job_id", "refolded_cif"])
        writer.writeheader()
        for row in reader:
            seq, seed = row["edited_sequence"], int(row["seed"])
            if not 0 <= seed < 2**32:
                raise ValueError("AF3 requires an unsigned 32-bit seed")
            job = "fm_" + digest(seq, seed)[:32]
            path = args.results / job / f"{job}_model.cif"
            if job not in seen:
                seen.add(job)
                data = {"name": job, "dialect": "alphafold3", "version": 1,
                        "modelSeeds": [seed],
                        "sequences": [{"protein": {"id": "A", "sequence": seq}}]}
                job_file = args.jobs / f"{job}.json"
                if args.require_results:
                    if json.loads(job_file.read_text()) != data:
                        raise ValueError(f"AF3 input does not match the edit manifest: {job_file}")
                    if not path.is_file() or path.stat().st_size == 0:
                        raise FileNotFoundError(path)
                else:
                    with output_file(job_file) as handle:
                        json.dump(data, handle)
            writer.writerow(dict(row, job_id=job, refolded_cif=str(path.resolve())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("edits", type=Path)
    p.add_argument("mapping", type=Path)
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--require-results", action="store_true")
    main(p.parse_args())
