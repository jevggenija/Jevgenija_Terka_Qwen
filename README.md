# LoRA Training Project

## Project Description
This repository contains the code and configuration to train a **LoRA (Low-Rank Adaptation) model** for image generation.  
LoRA enables efficient fine-tuning of large pre-trained models by updating only a small set of low-rank weights, reducing computational and memory requirements.  

This project specifically focuses on generating images inspired by the **Zdzisław Beksiński** art style.

---

## 1. Installation

### Requirements
- **Python**: 3.10  
- **PyTorch**: 2.2.0  
- **CUDA**: 12.1.1 (devel) on Ubuntu 22.04  

**Recommended System Configuration:**  
| Component | Specification | Notes |
|-----------|---------------|-------|
| RAM       | 151 GB        | Eliminates risk of Out-of-Memory (OOM) errors |
| vCPU      | 9             | Sufficient for smooth data processing |
| Disk      | 400 GB        | Enough space for OS, container, base model, dataset, and checkpoints |
| GPU       | NVIDIA L40 48 GB VRAM | Ideal for training large LoRA models like Flux or Qwen |

---
## 2. Workspace Setup on RunPod

1. **Install ComfyUI**  
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git

2. **Install ComfyUI**   Create a Python virtual environment
   python -m venv venv
3. Activate the environment

source venv/bin/activate



4. 
