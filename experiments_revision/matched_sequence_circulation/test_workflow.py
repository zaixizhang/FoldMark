"""CPU regression checks. Fixtures contain no study measurements."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from common import METHODS, OPS, IDENTITIES
from step1_edit import run as edit
from step2_af3 import main as af3
from step3_score import validate_foldmark_codes
from step4_figure import evaluate


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = SimpleNamespace(manifest=self.root / "inputs.csv", output=self.root / "edits.csv",
                                    seeds=[23, 31], operators=["blosum62"])
        seq = "ACDEFGHIKLMNPQRSTVWY" + "A" * 5
        self.rows = [dict(sample_id="protein-1", method=m, split="test", label=label,
                          sequence=seq, detector_context="01"*16)
                     for m in METHODS for label in ("0", "1")]

    @staticmethod
    def write(path, rows):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_preflight(self):
        self.write(self.args.manifest, self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "Missing paired"):
            edit(self.args)
        self.assertFalse(self.args.output.exists())
        self.write(self.args.manifest, self.rows)
        self.args.seeds = [-1]
        with self.assertRaisesRegex(ValueError, "unsigned"):
            edit(self.args)

    def test_pairing_and_af3_verification(self):
        self.write(self.args.manifest, self.rows)
        edit(self.args)
        with self.args.output.open() as f:
            edited = list(csv.DictReader(f))
        for row in edited:
            mutated = {i for i, (x, y) in enumerate(zip(row["sequence"], row["edited_sequence"]), 1)
                       if x != y}
            self.assertEqual(mutated, {int(i) for i in row["positions"].split()})
            # Length 25: 90% identity requires rounding 2.5 edits to 3.
            if row["identity"] == "90":
                self.assertEqual(len(mutated), 3)
        for seed in self.args.seeds:
            paired = [r["positions"] for r in edited if r["seed"] == str(seed) and r["identity"] == "90"]
            self.assertEqual(len(set(paired)), 1)
        validate_foldmark_codes(self.args.output)
        args = SimpleNamespace(edits=self.args.output, mapping=self.root/"mapping.csv",
                               jobs=self.root/"jobs", results=self.root/"results", require_results=False)
        af3(args)
        jobs = sorted(args.jobs.glob("*.json"))
        self.assertLess(len(jobs), len(edited))
        original_mapping = args.mapping.read_bytes()
        args.require_results = True
        with self.assertRaises(FileNotFoundError):
            af3(args)
        self.assertEqual(args.mapping.read_bytes(), original_mapping)
        for f in jobs:
            output = args.results / f.stem / (f.stem + "_model.cif")
            output.parent.mkdir(parents=True)
            output.write_text("fixture for file presence only")
        af3(args)
        changed = json.loads(jobs[0].read_text())
        changed["modelSeeds"] = [999]
        jobs[0].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "does not match"):
            af3(args)
        self.assertEqual(json.loads(jobs[0].read_text()), changed)

    def test_heldout_operating_point(self):
        mapping, scores = [], []
        # At target 1/5, the tied second-highest calibration value is 1:
        # strict > 1 accepts exactly one calibration negative. Test FPR is 1/2.
        examples = [("calibration", str(i), 0, v) for i, v in enumerate([2, 1, 1, 0, 0])]
        examples += [("test", "a", 0, 2), ("test", "b", 0, 1),
                     ("test", "a", 1, 2), ("test", "b", 1, 2)]
        for run in (1, 2):
            for method in METHODS:
                for op in OPS:
                    for identity in IDENTITIES:
                        for split, sid, label, value in examples:
                            rid = str(len(mapping))
                            mapping.append(dict(run_id=run, seed=run, method=method, operation=op,
                                                identity=identity, split=split, sample_id=sid,
                                                label=label, row_id=rid))
                            score = 1 if run == 2 and label else value
                            scores.append(dict(row_id=rid, score=score, scTM=0.9))
        self.write(self.args.manifest, mapping)
        self.write(self.args.output, scores)
        runs, decisions = evaluate(self.args.manifest, [self.args.output], 2, 2, 0.2)
        self.assertEqual(set(runs.threshold), {1})
        self.assertEqual(set(runs.calibration_fpr), {0.2})
        self.assertEqual(set(runs.test_fpr), {0.5})
        self.assertEqual(set(runs[runs.run_id == "1"].tpr), {1})
        self.assertEqual(set(runs[runs.run_id == "2"].tpr), {0})
        self.assertEqual(set(decisions.split), {"test"})

    def test_mismatched_query_code(self):
        rows = [dict(sample_id="p", run_id=1, method="foldmark", label=l,
                     detector_context=b*32) for l, b in ((0, "0"), (1, "1"))]
        self.write(self.args.manifest, rows)
        with self.assertRaisesRegex(ValueError, "same registered code"):
            validate_foldmark_codes(self.args.manifest)


if __name__ == "__main__":
    unittest.main()
