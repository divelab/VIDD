# Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design (VIDD)

This repository contains the official implementation of paper [Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design](https://arxiv.org/pdf/2507.00445), including DNA sequence design, protein sequence design and molecule design.


## Overview

We address the problem of fine-tuning diffusion models for reward-guided generation in biomolecular design. 
We propose an iterative distillation-based fine-tuning framework that casts the problem as policy distillation.

## Installation

Run the installation script to set up the environment:

```bash
bash install.sh
```

**Note**: evodiff requires Python ≤ 3.9 for compatibility.

## Quick Start


### Protein Optimization

Protein experiments include:
- **Binding optimization**: Optimize sequences for binding to target proteins (PD-L1, IFNAR2)
- **Secondary structure**: Optimize for maximizing β-sheet

Example scripts:
- `scripts/protein_binder_PD_L1.sh` - PD-L1 binding optimization
- `scripts/protein_binder_IFNAR2.sh` - IFNAR2 binding optimization  
- `scripts/protein_ss.sh` - Secondary structure optimization



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{su2025iterative,
  title={Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design},
  author={Su, Xingyu and Li, Xiner and Uehara, Masatoshi and Kim, Sunwoo and Zhao, Yulai and Scalia, Gabriele and Hajiramezanali, Ehsan and Biancalani, Tommaso and Zhi, Degui and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2507.00445},
  year={2025}
}
```

