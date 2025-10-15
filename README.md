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
- Ensure the Python virtual environment is activated before running any scripts
- Large datasets and models require sufficient disk space and memory to prevent crashes
- Recommended hardware: NVIDIA L40 GPU, 151 GB RAM, 400 GB disk
## 4. References
- Image Models
   - SD1.x,[unCLIP](https://comfyanonymous.github.io/ComfyUI_examples/unclip/))
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager GitHub](https://github.com/Comfy-Org/ComfyUI-Manager)

    
# LoRA Training Summary

## 1. Dataset
- **Source:** `dataset_Lora_test` (loaded from a folder)
- **Processing:**
  - Images loaded using `LoadImageTextSetFromFolderNode`.
  - Resizing: Original size (width and height = -1).
  - CLIP conditioning applied for training.

## 2. Base Model
- **Checkpoint:** `sd_xl_base_1.0.safetensors`
- **Components Loaded:**
  - **MODEL:** Base model weights
  - **CLIP:** For text conditioning
  - **VAE:** For latent space encoding

## 3. Latent Encoding
- **Node Used:** `VAEEncode`
- **Input:** Dataset images
- **Output:** Latents for LoRA training

## 4. LoRA Training Parameters
- **Node Used:** `TrainLoraNode`
- **Batch Size:** 1
- **Gradient Accumulation Steps:** 8
- **Training Steps:** 1000
- **Learning Rate:** 0.0003
- **Optimizer:** AdamW
- **Loss Function:** MSE (Mean Squared Error)
- **Seed:** 550337636151787
- **Rank:** 8
- **Training Data Type:** `bf16`
- **LoRA Data Type:** `bf16`
- **Algorithm:** LoRA
- **Gradient Checkpointing:** Enabled
- **Existing LoRA:** None

## 5. Outputs
- **LoRA Model:** Saved at `loras/ChangedDataset`
- **Loss Graph:** Saved as `loss_graph`
- **Preview Image:** Displayed for quick inspection

## 6. Workflow Summary
1. Load dataset images with text captions.
2. Load base model checkpoint along with CLIP and VAE.
3. Encode images into latent space.
4. Train LoRA using encoded latents and positive conditioning.
5. Save trained LoRA and generate a loss graph for monitoring.


# 🎨 Dataset Overview

My image dataset consisted of **149 mixed paintings**, each paired with a **positive prompt**.  
Approximately **30 out of 149 images** included light scene descriptions — still focused primarily on **artistic style** rather than narrative content.

To enhance stylistic diversity and strengthen **LoRA learning**, each image was associated with **2–3 prompt variations**, phrased differently but describing the same overall aesthetic.

---

## 🧩 Examples

### Example 1 — Abstract (“pure style”)
```bash
<mojstylbeksinskiego> <mojstylbeksinskiego>, (masterpiece, best quality, ultra detailed), dystopian surrealism, painterly brush strokes, oil painting on hardboard, muted tones, dramatic lighting, surreal organic shapes, elongated forms, expression of despair and transcendence, atmospheric perspective

