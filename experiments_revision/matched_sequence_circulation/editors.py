"""Matched-position sequence editing for single-chain proteins."""
import random
import sys
from functools import lru_cache
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


@lru_cache(maxsize=8)
def backbone(pdb, chain):
    from Bio.PDB import PDBParser, Polypeptide
    structure = PDBParser(QUIET=True).get_structure("backbone", pdb)
    residues = [r for r in structure[0][chain] if r.id[0] == " "]
    sequence = "".join(Polypeptide.protein_letters_3to1[r.resname] for r in residues)
    coords = {f"{atom}_chain_{chain}": [r[atom].coord.tolist() for r in residues]
              for atom in ("N", "CA", "C", "O")}
    return {"name": "target", "num_of_chains": 1, "seq": sequence,
            "seq_chain_" + chain: sequence, "coords_chain_" + chain: coords}


class MpnnEditor:
    """Load vanilla ProteinMPNN once. Sample with a fresh paired seed per edit."""

    def __init__(self, repo, name, device, temperature):
        import torch
        sys.path.insert(0, str(Path(repo).resolve()))
        from protein_mpnn_utils import ProteinMPNN, tied_featurize
        self.featurize = tied_featurize
        self.torch = torch
        self.device = torch.device(device)
        self.temperature = temperature
        weights = torch.load(Path(repo) / "vanilla_model_weights" / f"{name}.pt",
                             map_location=self.device)
        self.model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                                 augment_eps=0.0, k_neighbors=weights["num_edges"])
        self.model.load_state_dict(weights["model_state_dict"])
        self.model.to(self.device).eval()

    def edit(self, sequence, sites, seed, pdb, chain):
        import numpy as np
        torch = self.torch
        target = backbone(pdb, chain)
        if target["seq"] != sequence:
            raise ValueError("MPNN backbone and circulated sequence differ")
        mutable = set(sites)
        fixed = [i+1 for i in range(len(sequence)) if i not in mutable]
        omit = [[[i+1], sequence[i]] for i in sites]
        # This follows the feature tuple returned by vanilla protein_mpnn_utils.tied_featurize.
        features = self.featurize([target], self.device, {"target": ([chain], [])},
                                  {"target": {chain: fixed}}, {"target": {chain: omit}})
        x, tokens, mask, _, design, encoding = features[:6]
        design_sites, excluded, residue_idx = features[10:13]
        bias_by_res = features[18]
        if not bool(mask.all()):
            raise ValueError("Incomplete backbone coordinates")
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = self.model.sample(
                x, torch.randn_like(design), tokens, design, encoding, residue_idx,
                mask=mask, temperature=self.temperature, chain_M_pos=design_sites,
                omit_AA_mask=excluded, bias_by_res=bias_by_res,
                omit_AAs_np=np.array([0]*20 + [1], dtype=np.float32),
                bias_AAs_np=np.zeros(21, dtype=np.float32))
        return "".join((AA + "X")[i] for i in result["S"][0].tolist())
