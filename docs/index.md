# UmeAiRT Toolkit

A modern, block-based node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — designed for clean workflows, auto-downloading model bundles, and pipeline-driven generation.

## Highlights

- **47 Nodes** organized into logical blocks (Loaders, Settings, Sampler, Post-Process, Utilities)
- **Block Architecture** — connect model bundles, settings, and prompts into a single pipeline
- **Auto-Download** — select a model family + version, and the Bundle Loader downloads missing files automatically
- **Multi-Architecture** — supports SD 1.5, SDXL, FLUX, Z-IMAGE (Lumina2), and fragmented (Diffusers) models
- **GGUF Support** — load quantized models directly (Q4, Q8, FP8) for low-VRAM setups
- **Built-in Post-Processing** — UltimateSD Upscale, SeedVR2 Upscale, FaceDetailer, Detailer Daemon

## Node Catalog (47 nodes)

| Category | Nodes | Description |
|----------|-------|-------------|
| [Loaders](nodes/index.md#loaders) | 7 | Checkpoint, FLUX, Z-IMG, Fragmented, Bundle, Multi-LoRA |
| [Generation](nodes/index.md#generation) | 6 | Settings, Image Loader/Process, LoRA Blocks, ControlNet, Prompts |
| [Sampling](nodes/index.md#sampling--post-process) | 5 | KSampler, UltimateSD Upscale, SeedVR2 Upscale |
| [FaceDetailer](nodes/index.md#sampling--post-process) | 4 | FaceDetailer, Detailer Daemon, BBOX Loader |
| [Image Output](nodes/index.md#image-output) | 3 | Source Image, Inpaint Composite, Image Saver |
| [Pack/Unpack](nodes/pack-unpack.md) | 13 | Interoperability nodes for bundle ↔ primitive conversion |
| [Utilities](nodes/index.md#utilities) | 5 | Label, Downloader, Log Viewer, Health Check, Signature |

## Quick Start

See [Getting Started](getting-started.md) for installation instructions.

<!-- TODO: Screenshot — Full workflow overview showing the block architecture -->
<!-- PLACEHOLDER: Add a screenshot of a complete workflow with Loader → Settings → Sampler → Upscale → Saver -->
