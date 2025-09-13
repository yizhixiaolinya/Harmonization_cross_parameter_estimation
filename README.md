# MRI Harmonization Project
> **Unified MRI Harmonization with Dual Image–Parameter Conditioning and Scanner-Aware Semantic Constraints**

## 📖 Overview
This repository will host the public release of our **Unified MRI Harmonization** framework, designed to mitigate inter-scanner and inter-protocol variability in large multi-site neuroimaging cohorts.  
The method integrates **dual image–parameter conditioning** and **scanner-aware semantic constraints** to translate MRI data across vendor styles (e.g., Siemens ↔ GE ↔ Philips) while preserving anatomical fidelity and quantitative integrity.

### Key Features (planned)
- **Parameter-Aware Generator**: Conditions on TR, TE, TI, FA, and voxel size for protocol-specific style transfer.  
- **Cross-Vendor Adaptation**: Supports any-to-any scanner/protocol harmonization.  
- **High-Fidelity Reconstruction**: Optimized for PSNR/SSIM and consistent imaging-parameter alignment.

## 🧠 Motivation
Multi-site neuroimaging enables large-scale studies of aging and neurodegenerative disease, but scanner hardware, acquisition protocols, and software upgrades create systematic intensity and contrast differences that hinder reproducibility and model generalization.  
Our harmonization framework addresses these issues to deliver:
- **Robust downstream analysis** (e.g., Alzheimer’s disease classification, lesion detection).  
- **Improved cross-site model performance** in both research and clinical workflows.

## 🚧 Project Status
The codebase is under **active re-organization and documentation**.  
- ✅ Core algorithms and training pipelines are complete and validated internally.  
- 📝 Public-facing modules and tutorials are being cleaned and modularized.

> **Estimated public release:** _TBA_.  
Watch or star the repository to receive updates.

## 🗂️ Planned Repository Layout
