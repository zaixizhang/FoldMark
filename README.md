# FoldMark: Protecting Protein Generative Models with Watermarking
<div align=center><img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/foldmark.png" width="202"/></div>

In the github repo, we apply FoldMark to [FrameFlow](https://github.com/microsoft/protein-frame-flow) as an example.

## Installation

```bash
# Conda environment with dependencies.
conda env create -f foldmark.yml

# Activate environment
conda activate fm

# Manually need to install torch-scatter.
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install local package.
# Current directory should be FoldMark/
pip install -e .
```

## Wandb

Our training relies on logging with wandb. Log in to Wandb and make an account.
Authorize Wandb [here](https://wandb.ai/authorize).

## Data

Download preprocessed SCOPe dataset (~280MB) hosted on dropbox: [link](https://www.dropbox.com/scl/fi/b8l0bqowi96hl21ycsmht/preprocessed_scope.tar.gz?rlkey=0h7uulr7ioyvzlap6a0rwpx0n&dl=0).

Other datasets are also possible to train on using the `data/process_pdb_files.py` script.
However, we currently do not support other datasets.

```bash
# Expand tar file.
tar -xvzf preprocessed_scope.tar.gz
rm preprocessed_scope.tar.gz
```
Your directory should now look like this 
```
├── analysis
├── build
├── configs
├── data
├── experiments
├── media
├── models
├── openfold
├── preprocessed
└── weights
```

## Pretrain

```bash
python -W ignore experiments/pretrain.py
```

## Pretrain

```bash
python -W ignore experiments/finetune.py
```

## Acknowledgements

We thank the opensource codes from [WaDiff](https://github.com/rmin2000/WaDiff) , [AquaLoRA](https://github.com/Georgefwt/AquaLoRA) and [openfold](https://github.com/aqlaboratory/openfold) .

## Reference
```
@article{zhang2024foldmark,
  title={FoldMark: Protecting Protein Generative Models with Watermarking},
  author={Zhang, Zaixi and Jin, Ruofan and Fu, Kaidi and Cong, Le and Zitnik, Marinka and Wang, Mengdi},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```





