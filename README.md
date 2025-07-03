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

FoldMark is a novel watermarking framework for protein generative models that embeds user-specific data across protein structures. It:
- Leverages evolutionary principles to adaptively embed watermarks (higher capacity in flexible regions, minimal disruption in conserved areas)
- Maintains structural quality (>0.9 scTM scores) while achieving >95% watermark bit accuracy at 32 bits
- Enables tracking of up to 1 million users and detection of unauthorized model training (even with only 30% watermarked data)
- Works with leading models like AlphaFold3, ESMFold, RFDiffusion, and RFDiffusionAA
- Withstands post-processing and adaptive attacks, offering a generalized solution for ethical protein AI deployment

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





