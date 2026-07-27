#!/usr/bin/env python3
"""
Post-processing operators: Amber relaxation, Rosetta FastRelax, NGK loop modeling
==================================================================================

Each operator takes a PDB path and writes a post-processed PDB, using **default
settings** for the corresponding tool — the point of the experiment is to test
what a routine, out-of-the-box refinement does to the watermark, not a tuned
variant.

Bound-state handling
--------------------
All operators run on the **complex as given** (protein + ligand, or
protein + antibody), so the refinement sees the binding partner and its
influence on the protein backbone is included in the result.  Two caveats that
matter for interpreting the numbers:

  · `amber`   — the OpenFold/AlphaFold Amber relaxer models **protein atoms
                only**.  Non-protein chains and HETATM records are stripped
                before minimisation and re-attached unchanged afterwards, so
                the ligand acts as a fixed context rather than a relaxed one.
                Use `--amber_backend openmm_complex` to instead relax the whole
                complex with an explicit ligand parameterisation (requires
                OpenFF/GAFF; see README).
  · `rosetta` — ligands need Rosetta params.  Pass `--params a.params b.params`
                or rely on `-load_PDB_components` for standard components.

The operators intentionally do not renumber or reorder residues: the FoldMark
decoder in Step 2 relies on residue correspondence with the input structure.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

METHODS = ["none", "amber", "rosetta_fastrelax", "rosetta_ngk"]

# Display names, matching the labels used in the revision figures.
METHOD_LABELS = {
    "none": "None",
    "amber": "Amber Relaxation",
    "rosetta_fastrelax": "Rosetta FastRelax",
    "rosetta_ngk": "NGK loop modeling",
}


@dataclass
class PostProcessResult:
    method: str
    out_pdb: Path
    ok: bool
    message: str = ""


# ── helpers ───────────────────────────────────────────────────────────────────

def split_protein_and_het(pdb_path: Path) -> tuple[list[str], list[str]]:
    """Split a PDB into (protein ATOM lines, HETATM/other-chain lines)."""
    protein_lines, het_lines = [], []
    for line in pdb_path.read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            protein_lines.append(line)
        elif line.startswith("HETATM"):
            het_lines.append(line)
        elif line.startswith(("TER", "END")):
            continue
        else:
            protein_lines.append(line)
    return protein_lines, het_lines


# ── none (baseline) ───────────────────────────────────────────────────────────

def run_none(in_pdb: Path, out_pdb: Path, **_) -> PostProcessResult:
    """Baseline: the unprocessed structure, copied so downstream code is uniform."""
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(in_pdb, out_pdb)
    return PostProcessResult("none", out_pdb, True)


# ── Amber relaxation ──────────────────────────────────────────────────────────

def run_amber(
    in_pdb: Path,
    out_pdb: Path,
    use_gpu: bool = False,
    max_iterations: int = 0,
    tolerance: float = 2.39,
    stiffness: float = 10.0,
    max_outer_iterations: int = 3,
    **_,
) -> PostProcessResult:
    """
    Amber relaxation via the OpenFold relaxer bundled with this repo
    (`openfold/np/relax/relax.py`), at its AlphaFold default settings.

    Non-protein records are held out of the minimisation and re-attached, so
    the binding partner is preserved in the output coordinates.
    """
    from openfold.np import protein as of_protein
    from openfold.np.relax import relax as of_relax

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    protein_lines, het_lines = split_protein_and_het(in_pdb)

    try:
        prot = of_protein.from_pdb_string("".join(protein_lines))
        relaxer = of_relax.AmberRelaxation(
            max_iterations=max_iterations,
            tolerance=tolerance,
            stiffness=stiffness,
            exclude_residues=[],
            max_outer_iterations=max_outer_iterations,
            use_gpu=use_gpu,
        )
        relaxed_pdb, _debug, _violations = relaxer.process(prot=prot)
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        return PostProcessResult("amber", out_pdb, False, f"{type(exc).__name__}: {exc}")

    body = [ln for ln in relaxed_pdb.splitlines(keepends=True)
            if not ln.startswith("END")]
    out_pdb.write_text("".join(body + het_lines) + "END\n")
    return PostProcessResult("amber", out_pdb, True)


# ── Rosetta ───────────────────────────────────────────────────────────────────

_PYROSETTA_READY = False


def init_pyrosetta(params: list[str] | None = None, extra_flags: str = "") -> None:
    """Initialise PyRosetta once per process."""
    global _PYROSETTA_READY
    if _PYROSETTA_READY:
        return
    import pyrosetta

    flags = ["-mute all", "-ignore_unrecognized_res false", "-load_PDB_components true"]
    if params:
        flags.append("-extra_res_fa " + " ".join(params))
    if extra_flags:
        flags.append(extra_flags)
    pyrosetta.init(" ".join(flags))
    _PYROSETTA_READY = True


def run_rosetta_fastrelax(
    in_pdb: Path,
    out_pdb: Path,
    params: list[str] | None = None,
    scorefxn_name: str = "ref2015",
    **_,
) -> PostProcessResult:
    """
    Rosetta FastRelax with default settings, on the full complex.
    """
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax

        init_pyrosetta(params)
        pose = pyrosetta.pose_from_pdb(str(in_pdb))
        scorefxn = pyrosetta.create_score_function(scorefxn_name)

        relax = FastRelax()               # default script, default cycles
        relax.set_scorefxn(scorefxn)
        relax.apply(pose)

        pose.dump_pdb(str(out_pdb))
    except Exception as exc:  # noqa: BLE001
        return PostProcessResult("rosetta_fastrelax", out_pdb, False,
                                 f"{type(exc).__name__}: {exc}")
    return PostProcessResult("rosetta_fastrelax", out_pdb, True)


def run_rosetta_ngk(
    in_pdb: Path,
    out_pdb: Path,
    params: list[str] | None = None,
    loop_start: int | None = None,
    loop_end: int | None = None,
    scorefxn_name: str = "ref2015",
    **_,
) -> PostProcessResult:
    """
    Next-generation KIC (NGK) loop modeling — Stein & Kortemme (2013),
    implemented in Rosetta's `protocols::kinematic_closure` namespace.

    If no loop is given, the longest contiguous non-helix / non-sheet segment
    (by DSSP) is remodelled, which is the natural "defaults" choice for a
    structure with no annotated loop of interest.
    """
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyrosetta
        from pyrosetta.rosetta.core.scoring.dssp import Dssp
        from pyrosetta.rosetta.protocols.kinematic_closure import KicMover
        from pyrosetta.rosetta.protocols.loops import Loop, Loops

        init_pyrosetta(params)
        pose = pyrosetta.pose_from_pdb(str(in_pdb))
        scorefxn = pyrosetta.create_score_function(scorefxn_name)

        if loop_start is None or loop_end is None:
            Dssp(pose).insert_ss_into_pose(pose)
            ss = pose.secstruct()
            best, cur_start = (0, 0, 0), None
            for i, code in enumerate(ss, start=1):
                if code == "L":
                    cur_start = i if cur_start is None else cur_start
                else:
                    if cur_start is not None:
                        if i - cur_start > best[0]:
                            best = (i - cur_start, cur_start, i - 1)
                        cur_start = None
            if cur_start is not None and len(ss) + 1 - cur_start > best[0]:
                best = (len(ss) + 1 - cur_start, cur_start, len(ss))
            _, loop_start, loop_end = best
            if loop_end - loop_start < 3:
                return PostProcessResult(
                    "rosetta_ngk", out_pdb, False,
                    "no loop of >=4 residues found for NGK remodeling",
                )

        loop = Loop(loop_start, loop_end, (loop_start + loop_end) // 2)
        loops = Loops()
        loops.add_loop(loop)

        kic = KicMover()
        kic.set_loop(loop)
        kic.apply(pose)
        scorefxn(pose)

        pose.dump_pdb(str(out_pdb))
    except Exception as exc:  # noqa: BLE001
        return PostProcessResult("rosetta_ngk", out_pdb, False,
                                 f"{type(exc).__name__}: {exc}")
    return PostProcessResult("rosetta_ngk", out_pdb, True)


DISPATCH = {
    "none": run_none,
    "amber": run_amber,
    "rosetta_fastrelax": run_rosetta_fastrelax,
    "rosetta_ngk": run_rosetta_ngk,
}


def apply_method(method: str, in_pdb: Path, out_pdb: Path, **kwargs) -> PostProcessResult:
    if method not in DISPATCH:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return DISPATCH[method](in_pdb, out_pdb, **kwargs)
