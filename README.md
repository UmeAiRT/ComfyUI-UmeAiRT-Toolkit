# 🌌 ComfyUI UmeAiRT Toolkit

**A Wireless, Block-Based, and Aesthetic Toolkit for ComfyUI.**

Stop fighting with "noodle soup"! The UmeAiRT Toolkit provides two workflow paradigms:

- **Wireless Nodes**: Share global state for ultra-clean workflows
- **Block Nodes**: Self-contained bundles that connect via typed sockets

<!-- ![Workflow Example](https://github.com/user-attachments/assets/placeholder.png) -->

---

## ✨ Key Features

### 📡 Wireless Architecture

- **Global State**: Define inputs (Model, VAE, CLIP, Steps, CFG, etc.) once with "Setter" nodes
- **Autonomous Processors**: KSampler, Upscaler, FaceDetailer automatically fetch from global state
- **Zero Wires**: No need to drag wires across your canvas

### 🧱 Block Architecture

- **Self-Contained Bundles**: Each block outputs a typed bundle (Models, Settings, Prompts, LoRAs)
- **Clean Connections**: Just connect blocks together - each handles its own complexity
- **Hybrid Ready**: Blocks also update global state for Wireless compatibility

### 🎨 Custom Color Theme

Each node family has its own color for instant visual recognition:

- 🔵 **Blue**: Model Loaders
- 🟢 **Green**: Prompts
- 🟤 **Amber**: Settings
- 🟣 **Violet**: LoRAs
- ⬛ **Gray**: Samplers
- 🔷 **Pale Blue**: Upscale/Detailer
- 🔴 **Rust Red**: Image Loader
- 🩵 **Teal**: Image Saver

---

## 📦 Nodes Overview

### Wireless Nodes

| Category | Node | Description |
|:---|:---|:---|
| **Variables** | Input/Output nodes | Get/Set Steps, CFG, Seed, Sampler, etc. |
| **Loaders** | `Wireless Checkpoint Loader` | Load Model/CLIP/VAE wirelessly |
| **Loaders** | `Wireless Image Loader` | Load source image for Img2Img |
| **Samplers** | `Wireless KSampler` | Auto-detect Txt2Img/Img2Img mode |
| **Post-Processing** | `Wireless Ultimate Upscale` | USDU with wireless inputs |
| **Post-Processing** | `Wireless FaceDetailer` | Face enhancement with wireless inputs |
| **Post-Processing** | `Detailer Daemon` | Autonomous 2-Pass Refiner (Simple/Adv) |
| **Output** | `Wireless Image Saver` | Auto-path and metadata saving |

### Block Nodes

| Category | Node | Description |
|:---|:---|:---|
| **Models** | `Model Loader (Block)` | Checkpoint loader with bundle output |
| **Models** | `FLUX Loader (Block)` | UNET + Dual CLIP + VAE loader |
| **Generation** | `Generation Settings` | Width, Height, Steps, CFG, Seed bundle |
| **Generation** | `Prompt (Block)` | Positive/Negative prompt bundle |
| **LoRA** | `LoRA 1x/3x/5x/10x` | Stackable LoRA loaders |
| **Images** | `Image Loader (Block)` | Load image with mode selection |
| **Samplers** | `Block Sampler` | Full sampler with bundle inputs |
| **Post-Processing** | `Block Ultimate Upscale` | USDU with bundle inputs |
| **Post-Processing** | `Block FaceDetailer` | Face enhancement with bundle inputs |

### Utilities

| Node | Description |
|:---|:---|
| `Label` | Aesthetic sticky notes for documentation |
| `Debug` | Inspect current global wireless state |
| `Bbox Detector Loader` | Load face detection models |
| `Detailer Daemon` | Standalone Detail Daemon logic |

---

## 🚀 Installation

### Option A: ComfyUI Manager (Recommended)

1. Open **ComfyUI Manager**
2. Search for `UmeAiRT Toolkit`
3. Click **Install**

### Option B: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/UmeAiRT/ComfyUI-UmeAiRT-Toolkit.git
pip install -r ComfyUI-UmeAiRT-Toolkit/requirements.txt
```

---

## 🛠️ Usage

### Wireless Workflow

1. Place **Setter** nodes (Checkpoint, Settings, Prompts)
2. Place **Wireless KSampler** - it auto-fetches everything
3. For Img2Img: Add `Wireless Image Loader` with `denoise < 1.0`

### Block Workflow

1. Connect `Model Loader` → `Block Sampler`
2. Connect `Generation Settings` → `Block Sampler`
3. Connect `Prompt Block` → `Block Sampler`
4. Optionally chain `LoRA` blocks between Model and Sampler

---

## ❤️ Credits

Developed by **UmeAiRT Team**.  
License: MIT
