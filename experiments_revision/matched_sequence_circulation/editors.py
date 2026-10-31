"""Matched-position sequence editing for single-chain proteins."""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"

# All highest-scoring non-self canonical replacements in BLOSUM62 (ties retained).
BEST = dict(zip(AA, ("S", "A", "E", "DQ", "Y", "ANS", "Y", "V", "R", "IM",
                     "L", "DHS", "ADQEKST", "E", "K", "ANT", "S", "I", "Y", "F")))


def blosum(sequence, sites, seed):
    rng = random.Random(seed)
    residues = list(sequence)
    for i in sites:
        residues[i] = rng.choice(BEST[sequence[i]])
    return "".join(residues)


class EsmEditor:
    """Sequential masked conditional maximization at a prescribed set of sites."""

    def __init__(self, checkpoint, device):
        import esm
        import torch
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(checkpoint)
        self.model.eval().to(device)
        self.device = device
        self.torch = torch

    def edit(self, sequence, sites):
        if len(sequence) > 1022:
            raise ValueError("ESM2 reference editor supports sequences up to 1022 residues")
        alphabet = self.alphabet
        _, _, tokens = alphabet.get_batch_converter()([("protein", sequence)])
        tokens = tokens.to(self.device)
        edited = list(sequence)
        with self.torch.inference_mode():
            for i in sites:
                tokens[0, i + 1] = alphabet.mask_idx
                logits = self.model(tokens)["logits"][0, i + 1]
                allowed = [a for a in AA if a != sequence[i]]
                ids = [alphabet.get_idx(a) for a in allowed]
                winner = int(logits[ids].argmax())
                edited[i] = allowed[winner]
                tokens[0, i + 1] = ids[winner]
        return "".join(edited)


def mpnn(sequence, sites, seed, pdb, chain, mpnn_dir, temperature, model_name):
    """Fix all other positions; forbid each source residue at its edited site."""
    from Bio.PDB import PDBParser, Polypeptide
    structure = PDBParser(QUIET=True).get_structure("backbone", pdb)
    coords = {f"{atom}_chain_{chain}": [] for atom in ("N", "CA", "C", "O")}
    residues = [r for r in structure[0][chain] if r.id[0] == " "]
    native = "".join(Polypeptide.protein_letters_3to1[r.resname] for r in residues)
    if native != sequence:
        raise ValueError("MPNN input must be a backbone of the circulated sequence, in sequence order")
    for residue in residues:
        for atom in ("N", "CA", "C", "O"):
            coords[f"{atom}_chain_{chain}"].append(residue[atom].coord.tolist())
    target = {"name": "target", "num_of_chains": 1, "seq": sequence,
              "seq_chain_" + chain: sequence, "coords_chain_" + chain: coords}
    mutable = set(sites)
    fixed = [i + 1 for i in range(len(sequence)) if i not in mutable]
    omit = [[[i + 1], sequence[i]] for i in sites]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for name, record in [("input", target), ("fixed", {"target": {chain: fixed}}),
                             ("omit", {"target": {chain: omit}})]:
            (out / (name + ".jsonl")).write_text(json.dumps(record) + "\n")
        subprocess.run([sys.executable, str(Path(mpnn_dir) / "protein_mpnn_run.py"),
                        "--jsonl_path", str(out / "input.jsonl"), "--out_folder", tmp,
                        "--fixed_positions_jsonl", str(out / "fixed.jsonl"),
                        "--omit_AA_jsonl", str(out / "omit.jsonl"), "--model_name", model_name,
                        "--sampling_temp", str(temperature), "--seed", str(seed),
                        "--num_seq_per_target", "1", "--batch_size", "1"], check=True,
                       stdout=subprocess.DEVNULL)
        from Bio import SeqIO
        native_record, design = list(SeqIO.parse(out / "seqs/target.fa", "fasta"))
        if str(native_record.seq) != sequence:
            raise ValueError("MPNN output sequence and input sequence disagree")
        return str(design.seq)
