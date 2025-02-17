# Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion
![version](https://img.shields.io/badge/version-1.0.1-6395ED)
![version](https://img.shields.io/badge/license-MIT-9ACD32)
[![preprint](https://img.shields.io/badge/Preprint'25-EE4C2C)](https://arxiv.org/abs/2501.15393)
[![DASFAA](https://img.shields.io/badge/DASFAA-2025-B57EDC)](https://dasfaa2025.github.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Introduction
This is the PyTorch implementation of the paper DASFAA 2025: [Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion](https://arxiv.org/pdf/2501.15393). We propose a novel **D**iffusion-based **H**ierarchical **N**egative **S**ampling (**DHNS**) framework tailored for multimodal knowledge graph completion (MMKGC) tasks, which tackles the challenge of generating high-quality negative triples by leveraging a Diffusion-based Hierarchical Embedding Generation (DiffHEG) that progressively conditions on entities and relations as well as multimodal semantics. Furthermore, we develop a Negative Triple-Adaptive Training (NTAT) strategy that dynamically adjusts training margins associated with the hardness level of the synthesized negative triples, facilitating a more robust and effective learning procedure to distinguish between positive and negative triples.

## 🌈 An Overview of the DHNS Framework
![image](https://github.com/ngl567/DHNS/blob/main/framework-1.png)

## 💻 Installation
Create a conda environment with pytorch:  
```
conda create --name dhns_env python=3.8
source activate dhns_env
pip install -r requirements.txt
```

## 📁 Datasets and Pretrained Embeddings
We reuse three multimodal knowledge graph datasets namely DB15K, MKG-W and MKG-Y in the folder ./benchmarks/ along with [MMRNS](https://github.com/quqxui/MMRNS).  
The visual and textual embeddings of the MMKGs could be downloaded from this [Google Drive](https://drive.google.com/drive/folders/1UJSfnb8DEx2s-k8zaQx1fWUw5f45GBpI?usp=sharing). You can kindly put all the files of format .pth in the folder ./embeddings/.

## 🚀 Train and Test
In order to reproduce the results of DHNS model on the datasets, you can kindly run the following command with DistMult+DHNS on MKG-Y for an instance:  
```
python train_dhns.py
```

## 🤝 Citation
If you use the codes, please cite the following paper:
```
@misc{niu2025dhns,
  author        = {Guanglin Niu and
                   Xiaowei Zhang},
  title         = {Diffusion-based Hierarchical Negative Sampling for Multimodal Knowledge Graph Completion},
  archivePrefix = {arXiv},
  year          = {2025},
  eprint        = {2501.15393},
  primaryClass  = {cs.AI}
}
```
