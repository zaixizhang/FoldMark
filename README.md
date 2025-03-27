# FoldMark: Protecting Protein Generative Models with Watermarking
<div align=center><img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/foldmark.png" width="202"/></div>

<div align=center>

[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Zaixi/FoldMark)
[![Paper](https://img.shields.io/badge/📄-Paper-green)](https://www.biorxiv.org/content/10.1101/2024.03.10.584409v1)

</div>

## 🌟 Try Our Demo!

We've created an interactive demo on Hugging Face Spaces where you can:
- Input protein sequences and get watermarked structure predictions
- Compare watermarked vs. non-watermarked structures
- Visualize the differences in 3D
- Pretrained Checkpoints and Inference code

[Try the Demo →](https://huggingface.co/spaces/Zaixi/FoldMark)

## 🚀 Overview

FoldMark is a novel watermarking framework for protein structure prediction models. It enables:
- Robust watermark embedding in protein structures
- Minimal impact on prediction accuracy
- Protection against model theft and unauthorized use

## 🛠️ Installation

```bash
# Create and activate conda environment
conda env create -f foldmark.yml
conda activate fm

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Install local package
pip install -e .
```

## 📊 Training Pipeline

### Data Setup
1. Download preprocessed SCOPe dataset (~280MB):
   [Download Link](https://www.dropbox.com/scl/fi/b8l0bqowi96hl21ycsmht/preprocessed_scope.tar.gz?rlkey=0h7uulr7ioyvzlap6a0rwpx0n&dl=0)
2. Extract the data:
   ```bash
   tar -xvzf preprocessed_scope.tar.gz
   rm preprocessed_scope.tar.gz
   ```

### Training Steps
1. Pretrain the model:
   ```bash
   python -W ignore experiments/pretrain.py
   ```
2. Finetune with watermarking:
   ```bash
   python -W ignore experiments/finetune.py
   ```

## 📝 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{zhang2024foldmark,
  title={FoldMark: Protecting Protein Generative Models with Watermarking},
  author={Zhang, Zaixi and Jin, Ruofan and Fu, Kaidi and Cong, Le and Zitnik, Marinka and Wang, Mengdi},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## 🙏 Acknowledgments

We thank the following open-source projects for their valuable contributions:
- [WaDiff](https://github.com/rmin2000/WaDiff)
- [AquaLoRA](https://github.com/Georgefwt/AquaLoRA)
- [openfold](https://github.com/aqlaboratory/openfold)
- [Protenix](https://github.com/bytedance/Protenix)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

