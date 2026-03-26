# 🌌 ComfyUI UmeAiRT Toolkit

![CI](https://github.com/UmeAiRT/ComfyUI-UmeAiRT-Toolkit/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/UmeAiRT/ComfyUI-UmeAiRT-Toolkit/actions/workflows/docs.yml/badge.svg)

**A Block-Based, Pipeline-Driven Toolkit for ComfyUI.**

Stop fighting with "noodle soup"! The UmeAiRT Toolkit uses a **hub-and-spoke** architecture where typed bundles flow through a clean pipeline — from model loading to post-processing — with full interoperability with native ComfyUI nodes.

![Workflow Example](examples/screenshots/Z-IMG_ALL2IMG.png)

---

## ✨ Key Features

### 🧱 Block Architecture (Hub-and-Spoke)

- **Typed Bundles**: Loaders output `UME_BUNDLE` (model+clip+vae), settings output `UME_SETTINGS`, and the sampler creates a `UME_PIPELINE` that flows through the entire post-processing chain
- **GenerationContext**: A single `gen_pipe` object carries models, settings, prompts, and the generated image — no global state, no race conditions
- **Direct Prompting**: Connect `Positive/Negative` prompt editors directly to the Block Sampler

### 🔄 Full Interoperability (Pack/Unpack)

- **Pack Models Bundle**: Use any native or community loader → pack into `UME_BUNDLE` → feed the Block Sampler
- **Unpack Pipeline**: Extract IMAGE, MODEL, CLIP, VAE, prompts, settings from `UME_PIPELINE` → connect to any native node
- **Unpack Nodes**: Decompose any UME bundle type into standard ComfyUI types

### 🎨 Custom Colors & UI

- **Automatic Connection Colors**: Custom colors for UME types are injected into any active ComfyUI theme
- **Intelligent Resizing**: Prompt nodes maintain readable sizes in Nodes 2.0
- **Color-Coded Categories**:
  - 🔵 **Blue**: Model Loaders
  - 🟢 **Green**: Prompts
  - 🟤 **Amber**: Settings & ControlNet
  - 🟣 **Violet**: LoRAs
  - ⬛ **Gray**: Sampler
  - 🔵 **Teal**: Post-Processing

---

## 📦 Nodes Overview

### Block Nodes (Core Pipeline)

| Category | Node | Description |
|:---|:---|:---|
| **Models** | `Model Loader` | Checkpoint loader → `UME_BUNDLE` |
| **Models** | `Model Loader - FLUX` | UNET + Dual CLIP + VAE → `UME_BUNDLE` |
| **Models** | `Model Loader (Fragmented)` | Separate UNET/CLIP/VAE files → `UME_BUNDLE` |
| **Models** | `📦 Bundle Auto-Loader` | Select category + version, auto-download & load (aria2 accelerated) |
| **Models** | `Multi-LoRA Loader` | Apply up to 3 LoRAs to MODEL + CLIP |
| **Settings** | `Generation Settings` | Width, Height, Steps, CFG, Seed → `UME_SETTINGS` |
| **Prompts** | `Positive / Negative Prompt Input` | Multiline text editors with dynamic prompts |
| **LoRA** | `LoRA 1x/3x/5x/10x` | Stackable LoRA loaders → `UME_LORA_STACK` |
| **Image** | `Image Loader` | Load and prepare source images → `UME_IMAGE` |
| **Image** | `Image Process` | Configure mode, denoise, auto-resize → `UME_IMAGE` |
| **Sampler** | `KSampler` | Central hub — receives all bundles → `UME_PIPELINE` |

### Post-Processing (Pipeline-Aware)

| Node | Description |
|:---|:---|
| `UltimateSD Upscale` / `(Advanced)` | Tiled upscaling with pipeline context |
| `SeedVR2 Upscale` / `(Advanced)` | AI upscaler (bundled) |
| `FaceDetailer` / `(Advanced)` | Face enhancement with BBOX detection |
| `Detailer Daemon` / `(Advanced)` | Advanced detail enhancement |
| `Inpaint Composite` | Inpainting with pipeline awareness |
| `Image Saver` | Save with metadata preservation |

### Pack/Unpack (Interoperability)

| Node | Direction | Description |
|:---|:---|:---|
| `Pack Models Bundle` | Native → UME | MODEL + CLIP + VAE → `UME_BUNDLE` |
| `Unpack Pipeline` | UME → Native | `UME_PIPELINE` → IMAGE + all 14 fields |
| `Unpack Models Bundle` | UME → Native | `UME_BUNDLE` → MODEL, CLIP, VAE |
| `Unpack Image Bundle` | UME → Native | `UME_IMAGE` → IMAGE, MASK, mode, denoise |
| `Unpack Settings/Prompts` | UME → Native | Extract individual settings or prompt strings |

### Utilities

| Node | Description |
|:---|:---|
| `Label` | Visual annotation node for organizing workflows |
| `💾 Bundle Model Downloader` | Download model bundles from HuggingFace (aria2 accelerated, no VRAM load — ideal for RunPod/cloud) |
| `📜 UmeAiRT Log Viewer` | View toolkit activity directly on the canvas |
| `🩺 Health Check` | Validate dependencies and optimizations at startup |

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
| `vendor/aria2/` | [aria2](https://github.com/aria2/aria2) | Tatsuhiro Tsujikawa | GPL-2.0 |

---

## 🔒 Security

UmeAiRT Toolkit is audited for common vulnerabilities:

- **Path Traversal**: Patched in Image Saver with a `while` loop sanitizer + `os.path.abspath()` validation against the output directory
- **HTTP Timeouts**: All `urlopen` calls enforce `timeout=30/60` to prevent hanging on unresponsive servers
- **Token Safety**: HF tokens are read from environment/cache, never exposed in workflow JSON

---

## 🧪 Testing

140+ unit tests with GitHub Actions CI on Python 3.10–3.14:

```bash
# Run all tests locally
python run_tests.py

# Run with coverage
coverage run --source=modules run_tests.py
coverage report -m --skip-covered
```

CI enforces a **40% coverage floor** (`--fail-under=40`).

New nodes should include structural tests — copy `tests/_template_node_test.py`.

---

## 📖 Documentation

Full documentation with node reference (I/O tables, usage tips) is available at:

🔗 **[toolkit.umeai.art](https://toolkit.umeai.art/)** (or [GitHub Pages](https://umeairt.github.io/ComfyUI-UmeAiRT-Toolkit/))

```bash
# Preview docs locally
pip install mkdocs-material
mkdocs serve  # → http://127.0.0.1:8000
```

---

## ❤️ Credits

Developed by **UmeAiRT**.  
License: MIT

![Signature](assets/signature.png)
