# AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Project Overview

This project develops a deep learning based image restoration system for improving degraded semiconductor inspection images.

The system is designed to restore images affected by noise and low resolution by performing denoising and super-resolution in a single restoration pipeline.

The objective is to recover fine structural details and produce a higher-quality image that can support downstream semiconductor inspection and defect analysis.

---

## Problem Statement

Semiconductor inspection images can suffer from:

- Noise
- Low spatial resolution
- Loss of fine structural details
- Image degradation during acquisition

These degradations can make small structures and possible defects difficult to observe.

This project addresses the problem using a deep learning image restoration model.

---

## Proposed Solution

The proposed solution uses a convolutional deep learning network to learn the mapping between degraded low-resolution images and their corresponding high-resolution ground-truth images.

### Restoration Pipeline

Degraded Low-Resolution Image
        ↓
Preprocessing
        ↓
V6 Restoration Network
        ↓
Denoising + Detail Reconstruction + Super-Resolution
        ↓
Restored High-Resolution Image
        ↓
PSNR / SSIM / LPIPS Evaluation

---

## Model Architecture

The final V6 model uses:

- Model: V5RestorationNet
- Input channels: 1
- Output channels: 1
- Features: 80
- Residual/processing blocks: 24
- Framework: PyTorch
- Training epochs: 200
- Optimizer: AdamW
- Learning-rate schedule: Cosine decay
- Initial learning rate: 5e-5
- Final learning rate: approximately 1e-7
- Batch size: 8
- Automatic Mixed Precision (AMP)

The model performs image restoration and 2× spatial upscaling.

Input resolution:
128 × 128

Output resolution:
256 × 256

---

## Training

The model was trained using paired degraded and ground-truth images.

The training objective uses a composite loss consisting of:

- Charbonnier Loss
- Mean Squared Error (MSE) Loss
- L1 Loss
- SSIM Loss

This combination was selected to balance pixel-level accuracy, structural similarity and perceptual image quality.

---

## Hardware and Software

### Hardware

- NVIDIA GeForce RTX 3050 Laptop GPU
- 6 GB GPU memory

### Software

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- CUDA
- Automatic Mixed Precision (AMP)

---

## Final V6 Results

The final V6 training achieved:

| Metric | Best V6 Result |
|---|---:|
| PSNR | 28.8176 dB |
| SSIM | 0.7817 |
| Training Epochs | 200 |
| Training Time | approximately 613 minutes |

Best model checkpoint:

`best_model_v6.pth`

---

## Example Restoration Result

The model was evaluated on individual test samples by comparing:

1. Noisy Low-Resolution Input
2. V6 Restored Image
3. Ground Truth Image

The restoration output demonstrates recovery of higher-resolution structural information from the degraded input.

For example, sample `00710.npy` achieved:

- Bicubic PSNR: 30.40 dB
- V6 PSNR: 39.49 dB
- Bicubic SSIM: 0.7166
- V6 SSIM: 0.9647
- Bicubic LPIPS: 0.2076
- V6 LPIPS: 0.0842

This represents a substantial improvement over bicubic interpolation for that representative sample.

---

## Evaluation Metrics

### PSNR

Peak Signal-to-Noise Ratio measures pixel-level reconstruction quality.

Higher PSNR generally indicates better reconstruction.

### SSIM

Structural Similarity Index measures similarity in structural information between the restored and ground-truth images.

Higher SSIM indicates better structural preservation.

### LPIPS

Learned Perceptual Image Patch Similarity measures perceptual similarity between images.

Lower LPIPS indicates better perceptual similarity.

---

## Project Output

The project generates:

- Restored images
- Ground-truth comparison images
- Input/output comparison visualizations
- NumPy restoration outputs
- Model checkpoints
- Training history and evaluation results

---

## Current Implementation Status

### Completed

- Dataset preparation
- Image degradation/restoration pipeline
- V6 model development
- 200-epoch training
- GPU-based training
- Model checkpoint generation
- Image restoration inference
- PSNR evaluation
- SSIM evaluation
- LPIPS-based comparison
- Visual comparison with ground truth
- V6 performance evaluation

### Future Work

The current implementation is a GPU-based software prototype.

Future development can focus on:

- FPGA/edge-GPU deployment
- Real-time inference optimization
- Model quantization
- Hardware acceleration
- Integration with semiconductor inspection systems
- Evaluation on larger and more diverse inspection datasets

---

## Project Structure

```text
KLA/
│
├── code/
│   ├── model/
│   ├── training/
│   ├── prediction/
│   └── evaluation/
│
├── results/
│   └── model_v6/
│       ├── checkpoints/
│       └── outputs/
│
├── README.md
└── requirements.txt
