# 🌌 ComfyUI UmeAiRT Toolkit

**A Wireless, Block-Based, and Aesthetic Toolkit for ComfyUI.**

Stop fighting with "noodle soup"! The UmeAiRT Toolkit provides two workflow paradigms:

- **Wireless Nodes**: Share global state for ultra-clean workflows
- **Block Nodes**: Self-contained bundles and streamlined inputs for maximum flexibility

<!-- ![Workflow Example](https://github.com/user-attachments/assets/placeholder.png) -->

---

## ✨ Key Features

### 📡 Wireless Architecture

- **Global State**: Define inputs (Model, VAE, CLIP, Steps, CFG, etc.) once with "Setter" nodes
- **Autonomous Processors**: KSampler, Upscaler, FaceDetailer automatically fetch from global state
- **Zero Wires**: No need to drag wires across your canvas (Uses `modules/common.py` state sharing)

### 🧱 Block Architecture

- **Self-Contained Bundles**: Each block outputs a typed bundle (Models, Settings, LoRAs)
- **Direct Prompting**: Connect standard `STRING` inputs (Positive/Negative) directly to your Samplers
- **Hybrid Ready**: Blocks also update global state for Wireless compatibility

### 🎨 Custom Color Theme & UI

- **UmeAiRT Dark Theme**: Automatically installs an optimized color palette for better node/cable visibility
- **Intelligent Resizing**: Prompt nodes automatically maintain readable sizes in Nodes 2.0
- **Node Coloring**:
  - 🔵 **Blue**: Model Loaders
  - 🟢 **Green**: Prompts
  - 🟤 **Amber**: Settings
  - 🟣 **Violet**: LoRAs
  - ⬛ **Gray**: Samplers
  - 🔵 **Teal**: Post-Processing

---

## 📦 Nodes Overview

### Wireless Nodes

| Category | Node | Description |
|:---|:---|:---|
| **Variables** | `Positive/Negative Prompt Input` | Set wireless prompt strings (with Output sockets) |
| **Variables** | `Global Seed / Resolution` | Set global settings wirelessly |
| **Loaders** | `Wireless Checkpoint Loader` | Load Model/CLIP/VAE wirelessly |
| **Samplers** | `Wireless KSampler` | Auto-detect Txt2Img/Img2Img mode |
| **Post-Processing** | `Wireless Ultimate Upscale` | USDU with wireless inputs |
| **Post-Processing** | `Wireless SeedVR2 Upscale` | SeedVR2 AI upscaler (bundled) |
| **Post-Processing** | `Wireless FaceDetailer` | Face enhancement with wireless inputs |

### Block Nodes

| Category | Node | Description |
|:---|:---|:---|
| **Models** | `Model Loader (Block)` | Checkpoint loader with bundle output |
| **Models** | `FLUX Loader (Block)` | UNET + Dual CLIP + VAE loader |
| **Generation** | `Generation Settings` | Width, Height, Steps, CFG, Seed bundle |
| **LoRA** | `LoRA 1x/3x/5x/10x (Block)` | Stackable LoRA loaders |
| **Samplers** | `Block Sampler` | Full sampler with bundle and direct prompt inputs |
| **Post-Processing** | `Face Detailer (Block)` | Face enhancement with bundle inputs |

### Utilities

| Node | Description |
|:---|:---|
| `Label` | Aesthetic sticky notes for documentation |
| `Debug` | Inspect current global wireless state |
| `Unpack Nodes` | Extract individual data from bundles (Image, Settings, Tags, etc.) |
| `📜 UmeAiRT Log Viewer` | View toolkit activity directly on the canvas |

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

> **Note**: An optimized theme "UmeAiRT Dark" is automatically installed. Select it in ComfyUI Settings > Color Palette.

---

## 📜 Third-Party Code & Attribution

This toolkit bundles or adapts code from the following open-source projects. We are grateful to their authors.

| Component | Source | Author(s) | License |
|:---|:---|:---|:---|
| `seedvr2_core/vendor/` | [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) | numz | Apache-2.0 |
| `usdu_core/` | [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) | ssitu | GPL-3.0 |
| `image_saver_core/` | [ComfyUI-Image-Saver](https://github.com/alexopus/ComfyUI-Image-Saver) | alexopus | MIT |
| Detail Daemon logic | [sd-webui-detail-daemon](https://github.com/muerrilla/sd-webui-detail-daemon) | muerrilla | MIT |
| `facedetailer_core/` | Inspired by [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | ltdrdata | GPL-3.0 |
| `vendor/comfyui_gguf/` | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | City96 | Apache-2.0 |

---

## ❤️ Credits

Developed by **UmeAiRT Team**.  
License: MIT
