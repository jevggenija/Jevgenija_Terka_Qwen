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
   ```
2. **Create a Python virtual environment**
    ```bash
    python -m venv venv
    ```
3. **Activate the environment**
    ```bash
    source venv/bin/activate
    ```
4. **Install ComfyUI Manager**
    ```bash
    git clone https://github.com/Comfy-Org/ComfyUI-Manager
    ```
5. **Launch ComfyUI**
  - Close your terminal and open a new one

  - Activate the environment again:
    ```bash
    source venv/bin/activate
    ```
  - Navigate to ComfyUI folder:
    ```bash
    cd ComfyUI
    ```
  - Launch the web interface:
    ```bash
    python main.py --listen
    ```
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

    
# 🧠 LoRA Training Summary

This document summarizes the configuration and workflow used to train a **LoRA (Low-Rank Adaptation)** model using **ComfyUI v0.4**.

---

## 📂 Dataset Configuration

| Parameter | Value |
|------------|--------|
| **Source Folder** | `dataset_Lora_test` |
| **Node Used** | `LoadImageTextSetFromFolderNode` |
| **Resize Method** | None *(original resolution preserved)* |
| **Width / Height** | `-1 / -1` |
| **CLIP Conditioning** | Enabled |

**Process Overview:**  
Images and captions are loaded from the `dataset_Lora_test` folder.  
The dataset keeps its original resolution, and CLIP text conditioning is applied for effective text-to-image alignment.

---

## 🧩 Base Model

| Component | Value |
|------------|--------|
| **Checkpoint** | `sd_xl_base_1.0.safetensors` |
| **Node Used** | `CheckpointLoaderSimple` |
| **Outputs** | MODEL, CLIP, VAE |

**Description:**  
The **Stable Diffusion XL base model** provides:  
- **MODEL**: Core diffusion model weights  
- **CLIP**: Text encoder for conditioning  
- **VAE**: Encoder/decoder for latent space transformation  

---

## 🌀 Latent Encoding

| Parameter | Value |
|------------|--------|
| **Node Used** | `VAEEncode` |
| **Input** | Dataset images |
| **Output** | Latent representations for LoRA training |

**Process:**  
Each image from the dataset is encoded into latent space using the base model’s VAE.  
These latents are used as inputs during LoRA training.

---

## ⚙️ LoRA Training Configuration

| Setting | Value |
|----------|--------|
| **Node Used** | `TrainLoraNode` |
| **Batch Size** | `1` |
| **Grad Accumulation Steps** | `8` |
| **Training Steps** | `1000` |
| **Learning Rate** | `0.0003` |
| **Optimizer** | `AdamW` |
| **Loss Function** | `MSE` *(Mean Squared Error)* |
| **Seed** | `550337636151787` |
| **Rank** | `8` |
| **Training Data Type** | `bf16` |
| **LoRA Data Type** | `bf16` |
| **Algorithm** | `LoRA` |
| **Gradient Checkpointing** | ✅ Enabled |
| **Existing LoRA** | `None` |

**Notes:**  
- `bf16` precision reduces VRAM usage while maintaining stability.  
- **AdamW** optimizer is ideal for fine-tuning low-rank parameters.  
- Gradient checkpointing minimizes memory usage during backpropagation.

---

## 💾 Outputs

| Output | Description / Path |
|---------|--------------------|
| **LoRA Model** | `loras/ChangedDataset` |
| **Loss Graph** | `loss_graph` |
| **Preview Image** | Displayed via `PreviewImage` node |

---

## 🧭 Workflow Summary

1. **Load Dataset** → Import images and captions from `dataset_Lora_test`.  
2. **Load Base Model** → Initialize `sd_xl_base_1.0.safetensors` (MODEL, CLIP, VAE).  
3. **Encode Latents** → Convert dataset images into latent representations using the VAE.  
4. **Train LoRA** → Fine-tune LoRA adapter layers with defined hyperparameters.  
5. **Save Outputs** → Export trained LoRA weights, loss graph, and preview image.

---

## 🧾 Key Takeaways

- **ComfyUI v0.4** workflow provides a modular visual setup for LoRA training.  
- **bf16 precision** improves memory efficiency without quality loss.  
- **Rank 8 LoRA** offers a good trade-off between flexibility and model size.  
- The resulting LoRA (`loras/ChangedDataset`) can be merged or applied to compatible **SDXL** models for personalization or style transfer.

---

> 🧩 **Tip:** This data setup can be extend to obtain better results by chaining multiple training datasets, enabling CLIP skip, or integrating custom loss functions for domain-specific LoRAs.



# 🎨 Dataset Overview

My image dataset consisted of **149 mixed paintings**, each paired with a **positive prompt**.  
Approximately **30 out of 149 images** included light scene descriptions — still focused primarily on **artistic style** rather than narrative content.

To enhance stylistic diversity and strengthen **LoRA learning**, each image was associated with **2–3 prompt variations**, phrased differently but describing the same overall aesthetic.

---

## 🧩 Examples

### Example 1 — Abstract (“pure style”)
   ```bash
<mojstylbeksinskiego> <mojstylbeksinskiego>, (masterpiece, best quality, ultra detailed), dystopian surrealism, painterly brush strokes, oil painting on hardboard, muted tones, dramatic lighting, surreal organic shapes, elongated forms, expression of despair and transcendence, atmospheric perspective
   ```
### Example 2 — Scene-based (“landscape / composition”)
   ```bash
A humanoid figure with bandaged head walking through ruins, painted in <mojstylbeksinskiego> style, elongated anatomy, haunting atmosphere, soft oil textures, dark red sky, emotional tension, chiaroscuro shadows
   ```
### 💀 Example 3 — Emotional variation
   ```bash
<mojstylbeksinskiego> <mojstylbeksinskiego>, somber surrealism, expressive brush strokes, depiction of anguish, blurred figures, oil painting technique, heavy textures, cold desaturated tones
   ```
---

## 🗂️ File Structure Example
image_001_1.jpg → 001_1.txt
image_001_2.jpg → 001_2.txt
image_001_3.jpg → 001_3.txt

---

## 💡 Why It Works

- 🔹 **Token reinforcement** — Repeating `<mojstylbeksinskiego>` at the beginning helps anchor the unique token.  
- 🔹 **Light textual variation** — Small changes in phrasing teach the model to understand the range of the style rather than memorize fixed descriptions.  
- 🔹 **Tag permutation** — Mixing the order of tags (lighting, color, composition) provides CLIP with richer contextual signals.  
- 🔹 **Phrase order shuffling** — Moving color information toward the beginning or end helps CLIP better capture *conceptual* rather than *syntactic* connections.

Using the same token (`<mojstylbeksinskiego>`) across all paintings ensures that **LoRA recognizes it as a shared stylistic identity.**

---

## 🌌 “Quiet” Compositions

To maintain balance and prevent overfitting on dense, figure-heavy scenes, the dataset also included **1–2 calm compositions** — such as minimal landscapes or architectural fragments.

Each of these “quiet” images had **three gentle prompt variants**, focusing on **atmosphere and color harmony** rather than complex structures.

---

### Example (Quiet Scene – `041`)

#### `041_1.jpg`
   ```bash
<mojstylbeksinskiego>, masterpiece, best quality, ultra-detailed, 8k, sharp focus, dark cosmic seascape, surreal misty shore with sandy reflective beach, turbulent dark night sky blending into crashing waves, massive mountainous wave or fog-shrouded hills in background, deep cool blue and teal palette, scattered bright stars and celestial glow mixed into clouds and waves, dark abstract entity with a bright star inside (upper right), solitary dark bird in flight (left center), small crescent shape on sand, highly textured oil painting on canvas, haunting, mysterious, nocturnal:1.4
   ```
#### `041_2.jpg`
   ```bash
<mojstylbeksinskiego>, ultra-detailed masterpiece, best quality, 8k, sharp focus, dark cosmic seascape, misty surreal shoreline with wet reflective sand, crashing waves merging with turbulent night sky, massive mountainous wave or fog-covered hills in distance, celestial glow and scattered stars woven into clouds and waves, dark abstract entity containing bright star in upper right, solitary bird or seagull flying left of center, small crescent shape on sand, deep cool blue and teal palette, highly textured oil painting on canvas, haunting, nocturnal, mysterious:1.4
   ```
#### `041_3.jpg`
   ```bash
<mojstylbeksinskiego>, best quality, masterpiece, ultra-detailed 8k, sharp focus, dark cosmic seascape, surreal misty shore, sandy beach with reflective wet edge, turbulent dark sky blending with crashing waves, massive mountainous wave or fog-shrouded hills in background, scattered stars and celestial glow in clouds and waves, dark abstract entity with bright star (upper right), solitary dark bird in flight (left center), small crescent on sand, deep cool blue and teal palette, highly textured oil painting on canvas, haunting, mysterious, nocturnal:1.4
   ```
---

### 🧠 Dataset Balance Strategy

“Quiet” scenes were **interleaved between every 8–10 dense compositions**, ensuring that **LoRA learned to transfer stylistic features** without overtraining on complex multi-figure scenes.

---
![Preview of "Quiet" image](assets/041_1.jpg)
_Example of a “quiet” composition from the dataset (image 041)._
