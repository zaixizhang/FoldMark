"""Script for preprocessing PDB files."""

import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md


from data import utils as du
from data import parsers
from data import errors


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--pdb_dir',
    help='Path to directory with PDB files.',
    type=str)
parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=50)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,)
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def process_file(file_path: str, write_dir: str):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path[0]).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path[0]
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path[3])

    # Extract all chains
    
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)

    complex_feats_wt = du.concat_np_features(struct_feats, False)
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path[0])

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)
    complex_feats['aatype_wt'] = complex_feats_wt['aatype']

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    #metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    metadata['modeled_seq_len'] = len(modeled_idx)
    
    try:
        # MDtraj
        traj = md.load(file_path[0])
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        # os.remove(file_path[0])
    except Exception as e:
        # os.remove(file_path[0])
        raise errors.DataError(f'Mdtraj failed with error {e}')

    chain_dict['ss'] = pdb_ss[0]
    metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # Radius of gyration
    metadata['radius_gyration'] = pdb_dg[0]
    metadata['smiles'] = file_path[2]
    metadata['ddg'] = file_path[1]
    
    # Write features to pickles.
    if len(complex_aatype)!=metadata['modeled_seq_len'] or len(complex_feats['bb_mask'])!=metadata['modeled_seq_len']:
        return None
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')
            
            
def get_file_paths():
    df = pd.read_csv('/n/holyscratch01/mzitnik_lab/MdrDB/full/MdrDB_release_v1.0.2022.tsv', sep='\t', low_memory=False)
    path = '/n/holyscratch01/mzitnik_lab/zaixizhang/MdrDB/full/data1/Project/Resistance_Database/Coscmd/MdrDB_v1.0.2022/'
    data_list = []
    for i in range(len(df)):
        id = df.iloc[i]['SAMPLE_ID']
        type = df.iloc[i]['TYPE'].replace(" ", "_")
        if type not in ['Single_Substitution']:
            continue
        smiles = df.iloc[i]['SMILES']
        ddg = df.iloc[i]['DDG.EXP']
        pdbid = df.iloc[i]['PDB_ID']
        mutate = df.iloc[i]['MUTATION']
        wt = os.path.join(path, type, id, 'WT_'+pdbid+'.pdb')
        mt = os.path.join(path, type, id, 'MT_'+pdbid+'_'+mutate+'.pdb')
        if not os.path.exists(mt):
            mt =mt.replace("+", "_")
        if not os.path.exists(mt):
            print('error')
            continue
        data_list.append([mt, ddg, smiles, wt])
    return data_list


def main(args):
    write_dir = '/n/holyscratch01/mzitnik_lab/MdrDB/processed/'
    
    all_file_paths = get_file_paths()
    total_num_paths = len(all_file_paths)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    metadata_file_name = 'metadata_full.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    main(args)