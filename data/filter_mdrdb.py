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

csv_path = '/n/holyscratch01/mzitnik_lab/MdrDB/processed/metadata_full.csv'
pdb_csv = pd.read_csv(csv_path)

def is_length_match(row):
    processed_file_path = row['processed_path']
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)
    
    # 返回长度匹配的布尔值
    return len(processed_feats['aatype']) == row['modeled_seq_len']

# 应用函数于每一行，并创建一个布尔索引
length_match_mask = pdb_csv.apply(is_length_match, axis=1)

# 使用布尔索引来过滤DataFrame
filtered_pdb_csv = pdb_csv[length_match_mask]
filtered_pdb_csv.to_csv(csv_path, index=False)