# FoldMark: Safeguarding Protein Structure Generative Models with Distributional and Evolutionary Watermarking
<div align=center><img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/foldmark.png" width="202"/></div>

<div align=center>

[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/Zaixi/FoldMark)
[![Paper](https://img.shields.io/badge/📄-Paper-green)](https://www.biorxiv.org/content/10.1101/2024.10.23.619960v7)
[![Twitter](https://img.shields.io/badge/𝕏-Twitter-black)](https://x.com/ZaixiZhang/status/1937877504842158503)

</div>

## 📰 Media Coverage
- **Science**: [Built-in safeguards might stop AI from designing bioweapons](https://www.science.org/content/article/built-safeguards-might-stop-ai-designing-bioweapons)
- **Nature Biotechnology**: [Watermarking generative AI for protein structure](https://www.nature.com/articles/s41587-025-02650-8)
- **Princeton AI Lab**: [Deep Dive Series: Building Biosecurity Safeguards into AI for Science](https://blog.ai.princeton.edu/2025/06/26/deep-dive-series-building-biosecurity-safeguards-into-ai-for-science/)


## 🌟 Try Our Demo!

We've created an interactive demo on Hugging Face Spaces where you can:
- Input protein sequences and get watermarked structure predictions
- Compare watermarked vs. non-watermarked structures
- Visualize the differences in 3D
- Pretrained Checkpoints and Inference code

[Try the Demo →](https://huggingface.co/spaces/Zaixi/FoldMark)

## 🚀 Overview

FoldMark is a first-of-its-kind watermarking strategy designed to provide essential biosecurity safeguards for generative protein models against dual-use risks. It:
- **Balances Performance and Quality:** Employs distributional and evolutionary principles to embed watermarks while maintaining high-fidelity protein structures.
- **High Bit Accuracy:** Achieves over 95% watermark bit accuracy at 32 bits with minimal impact on structural integrity (maintaining >0.9 scTM scores).
- **Broad Compatibility:** Works seamlessly with leading models, including AlphaFold3, ESMFold, RFDiffusion, and RFDiffusionAA.
- **Robust User Tracing:** Capable of successfully tracing the source of a generated protein back to one of up to 1 million users.
- **Wet Lab Validated:** Successfully tested on redesigned EGFP and CRISPR-Cas13, which showed wildtype-level function (98% fluorescence, 95% editing efficiency) and >90% watermark detection, proving its practical utility.


## 📊 Results

### Structure Prediction with Watermarking
<div align=center>
<img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/Struct_pred.png" width="600"/>
</div>

### De Novo Protein Structure Design with Watermarking
<div align=center>
<img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/de_novo.png" width="600"/>
</div>

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
## 🔬 Wet Lab Verifications on GFP and Cas13 Redesign
<div align=center>
<img src="https://github.com/zaixizhang/FoldMark/blob/main/assets/foldmark_wetlab.png" width="600"/>
</div>

## 📖 Reproduction Tutorials

Step-by-step scripts to reproduce the eGFP and Cas13 wet-lab watermarking
experiments are provided in [`tutorials/`](tutorials/).

Each protein has **three scripts** covering the full pipeline:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `step1_watermarked_structure_prediction.py` | Run FoldMark-Protenix to obtain a watermarked backbone |
| 2 | `step2_proteinmpnn_inverse_folding.py` | Partial inverse folding with ProteinMPNN (100 sequences, T = 0.1) |
| 3 | `step3_esm2_ranking.py` | Score with ESM2-650M and export top constructs for synthesis |

**eGFP (PDB 4EUL)** — design regions: residues 15–40 and 160–190 (surface-exposed loops).
Chromophore residues and proton-wire residues are fixed. 12 constructs synthesised;
98% fluorescence and >90% watermark bit accuracy.

**Cas13 (PDB 7VTI, apo/inactive state)** — design region: residues 258–325 (helical lid).
HEPN catalytic dyads are fixed. Top constructs showed 95% editing efficiency and
over 90% watermark bit accuracy. 

```bash
# Quick start (run from the FoldMark HuggingFace Space root after installation)
python tutorials/egfp/step1_watermarked_structure_prediction.py
python tutorials/egfp/step2_proteinmpnn_inverse_folding.py --mpnn_dir ./ProteinMPNN
python tutorials/egfp/step3_esm2_ranking.py
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
- [Protenix](https://github.com/bytedance/Protenix)
- [FrameFlow](https://github.com/microsoft/protein-frame-flow)
- [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion)
- [RFDiffusion](https://github.com/RosettaCommons/RFdiffusion)
- [Boltz-1](https://github.com/openfold/boltzmann)
- [Chai-1](https://github.com/chaidiscovery/chai-lab)
- [RFDiffusionAA](https://github.com/baker-laboratory/rf_diffusion_all_atom)
- [WaDiff](https://github.com/rmin2000/WaDiff)
- [AquaLoRA](https://github.com/Georgefwt/AquaLoRA)
- [openfold](https://github.com/aqlaboratory/openfold)


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.





