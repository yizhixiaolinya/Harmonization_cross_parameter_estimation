# MRI Harmonization Project
> **Imaging-Parameter–Aware Multi-Site MRI Harmonization**

## 📖 Introduction
This repository hosts the upcoming implementation of our **MRI harmonization framework**, which aims to reduce inter-scanner and inter-protocol variability in multi-site neuroimaging studies.  
Our method leverages imaging-parameter–aware deep learning to translate MRI data between different scanner styles (e.g., Siemens ↔ GE ↔ Philips) while preserving anatomical fidelity and quantitative consistency.

Key features to be included:
- **Parameter-Aware Generator**: Utilizes TR, TE, TI, FA, and voxel size as conditioning for style transfer.  
- **Cross-Site Harmonization**: Supports any-to-any vendor/protocol adaptation.  
- **High-Fidelity Reconstruction**: Optimized for PSNR/SSIM and scanner-parameter alignment.

## 🧠 Motivation
Multi-site neuroimaging is essential for large-scale studies of aging and neurodegenerative disease.  
However, scanner hardware, acquisition protocols, and software upgrades introduce systematic intensity and contrast differences, which hinder reproducibility and model generalization.  
Our harmonization approach addresses these challenges, enabling:
- **Reliable downstream analysis** (e.g., AD classification, lesion detection).  
- **Improved cross-site model performance** in clinical and research settings.

## 🚧 Project Status
The codebase is currently under **active re-organization and documentation**.  
- ✅ Core algorithms and training pipelines are complete in internal experiments.  
- 📝 We are cleaning, modularizing, and preparing the code for public release.  

> **Estimated public release:** _To be announced_.  
Stay tuned by watching or starring the repository for updates.

## 🗂️ Planned Repository Structure
