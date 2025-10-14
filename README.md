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

2. **Create a Python virtual environment**
    ```bash
    python -m venv venv
    
3. **Activate the environment**
    ```bash
    source venv/bin/activate
    
4. **Install ComfyUI Manager**
    ```bash
    git clone https://github.com/Comfy-Org/ComfyUI-Manager

5. **Launch ComfyUI**
  - Close your terminal and open a new one

  - Activate the environment again:
    ```bash
    source venv/bin/activate
    
  - Navigate to ComfyUI folder:
    ```bash
    cd ComfyUI
    
  - Launch the web interface:
    ```bash
    python main.py --listen

  - Wait ~30 seconds until a local address with port 8188 appears, indicating ComfyUI is running
6. **Access ComfyUI from browser**
  - In RunPod, locate Direct TCP Port Mapping
  - Copy the Public IP and external port (e.g., "http://206.41.93.58:52271/")
  - Paste into your browser to open ComfyUI with ComfyUI Manager
7. **Load JSON models to generate new images using the Qwen model**

## 3. Notes
    
## 4. References
- Image Models
   - SD1.x,[unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager GitHub](https://github.com/Comfy-Org/ComfyUI-Manager)

    




4. 
