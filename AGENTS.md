# UmeAiRT Toolkit - Agent Development Guide

> Instructions for AI coding agents working on this project.
> For architecture details, see `/docs/codemaps/`.
>
> This file follows the [AGENTS.md](https://agents.md) standard.

## Project Overview

ComfyUI Custom Nodes toolkit with two node families:
1. **Block Nodes** (primary): Hub-and-spoke architecture using typed objects (`UME_BUNDLE`, `UME_SETTINGS`, `UME_PIPELINE`).
2. **Pipeline Nodes**: Post-processing and I/O nodes that operate on `GenerationContext` (UME_PIPELINE).

## Ecosystem (Sibling Projects on `Y:\`)

This project is part of a 6-project ecosystem. **Direct** relationships:

| Project | Relationship |
|---------|-------------|
| `ComfyUI-Workflows` | Workflows will be migrated to use this Toolkit's wireless nodes (Toolkit still in development — not yet integrated) |
| `ComfyUI-Auto_installer` | The installer auto-installs this Toolkit as a custom node via `custom_nodes.json` |
| `ComfyUI-UmeAiRT-Sync` | The Sync node distributes workflows that depend on this Toolkit |
| `UmeAiRT-NAS-Utils` | Orchestration hub — may run consistency checks against this project |

> ⚠️ **Impact awareness**: Renaming or removing a node class will break existing workflows in `ComfyUI-Workflows`. Always check workflow compatibility before modifying `NODE_CLASS_MAPPINGS`.

## Critical Conventions

### Block Architecture (Hub-and-Spoke)

The **BlockSampler** is the central hub. Side-input nodes feed into it, and the generated image flows through post-processing via the `UME_PIPELINE`.

```
Loader ──▶ UME_BUNDLE {model, clip, vae, model_name}
                │
Settings ──▶ UME_SETTINGS {width, height, steps, cfg, ...}
                │
Prompts ────────┤
LoRAs ──────────┤──▶ BlockSampler ──▶ UME_PIPELINE (GenerationContext)
Source Image ───┘                          │
                                           ├──▶ Post-Process nodes (read/write pipeline.image)
                                           └──▶ ImageSaver (reads pipeline.image)
```

**Key types:**

| Type | Content | Produced by |
|------|---------|-------------|
| `UME_BUNDLE` | `{model, clip, vae, model_name}` | All Loader nodes |
| `UME_SETTINGS` | `{width, height, steps, cfg, sampler_name, scheduler, seed}` | GenerationSettings |
| `UME_PIPELINE` | `GenerationContext` object (image + all context) | BlockSampler |
| `UME_IMAGE` | `{image, mask, mode, denoise, auto_resize}` | BlockImageLoader → BlockImageProcess |

**Rules:**
- Post-process nodes receive `UME_PIPELINE`, read `pipeline.image`, process, update `pipeline.image`, return `UME_PIPELINE`.
- Never create `GenerationContext` outside the `BlockSampler`.
- The `auto_resize` flag in `UME_IMAGE` is acted upon by the `BlockSampler` using `UME_SETTINGS` dimensions.



### Coding Standards

**Naming:**

- Class Names: `UmeAiRT_` prefix (e.g., `UmeAiRT_BlockSampler`).
- Display Names: Clear, user-friendly (e.g., "KSampler (Block)").
- Output names: `model_bundle` for loaders, `generation` for sampler/post-process.

**Registration:**

- All new nodes **MUST** be registered in `__init__.py` in two places:
    1. `NODE_CLASS_MAPPINGS`
    2. `NODE_DISPLAY_NAME_MAPPINGS`

### File Structure

- `modules/common.py`: `GenerationContext` class, `TypedDict` bundle types (`UmeBundle`, `UmeSettings`, `UmeImage`), shared helpers (`resize_tensor`, `encode_prompts`, `apply_outpaint_padding`).
- `modules/logger.py`: Standard logging utility.
- `modules/optimization_utils.py`: Environment and optimization checks.
- `modules/extra_samplers.py`: Custom KSampler algorithms (SA-Solver, RES Multistep).
- `modules/block_nodes.py`: Block Loaders (→ `UME_BUNDLE`), GenerationSettings (→ `UME_SETTINGS`), BlockSampler (→ `UME_PIPELINE`), Block post-processors, and BundleAutoLoader.
- `modules/logic_nodes.py`: Pipeline-aware Upscalers, Detailers, and Detail Daemon.
- `modules/image_nodes.py`: Image loading, processing, saving (pipeline-aware).
- `modules/model_nodes.py`: Multi-LoRA Loader.
- `modules/utils_nodes.py`: Labels, debuggers, Pack/Unpack interoperability nodes, Bundle Downloader.
- `__init__.py`: Registration and exposing nodes to ComfyUI.
- `web/`: Javascript extensions (UI tweaks, colors, Nodes 2.0 enforcements).
- `*/core/`: Integrated libraries (e.g., `usdu_core`, `seedvr2_core`).
- `vendor/comfyui_gguf/`: Vendored implementation of `ComfyUI-GGUF` for `.gguf` weight loading.

## UI & Styling (Node Colors)

Nodes are color-coded by category in `web/umeairt_colors.js`:

| Category | Color Family | Hex (Bg/Fg) | Examples |
|----------|--------------|-------------|----------|
| **Settings / Controls**   | Amber / Bronze | `#4A290B` / `#935116` | Generation Settings, Image Process, ControlNet |
| **Model / Files**         | Deep Blue      | `#0A2130` / `#154360` | Checkpoint Loader, VAE, CLIP |
| **Prompts**               | Dark Green     | `#0A2D19` / `#145A32` | Positive/Negative Prompt Input |
| **LoRA**                  | Violet         | `#25122D` / `#4A235A` | LoRA Stacks |
| **Samplers (Processors)** | Slate Gray     | `#1A252F` / `#2C3E50` | Block Sampler |
| **Post-Processing**       | Pale Blue / Teal | `#123851` / `#2471A3` | Ultimate Upscale, Face Detailer, Inpaint Composite |
| **Utilities**             | Dark Gray      | `#1A252F` / `#34495E` | Debug, Label, UmeAiRT Signature |
| **Image Inputs**          | Rust Red       | `#35160D` / `#6B2D1A` | Image Loaders |

**Connection colors:**

| Type | Color | Hex |
|------|-------|-----|
| `UME_BUNDLE` | Bright Blue | `#3498DB` |
| `UME_PIPELINE` | Teal | `#1ABC9C` |
| `UME_SETTINGS` | Amber/Copper | `#CD8B62` |
| `UME_IMAGE` | Orange/Brown | `#DC7633` |
| `UME_LORA_STACK` | Purple | `#9B59B6` |

## Project Maintenance & Stability Rules

To avoid regressions and maintain a stable, production-ready codebase, adhere strictly to the following rules:

1. **Dependency Synchronization**: Always update `pyproject.toml` instantly when adding a new package to `requirements.txt`. They must mirror each other to guarantee seamless node installation for users.
2. **Proper Exception Handling**: **NEVER** use bare exceptions (`except:` or `except: pass`). Always catch specific exceptions or use `except Exception as e:` and log the error via `log_node()` so failures are visible during debugging.
3. **Changelog Maintenance**: All notable modifications, bug fixes, or additions must be immediately documented in `CHANGELOG.md` following the *Keep a Changelog* format.

## Critical Files

| File | Notes |
|------|-------|
| `modules/common.py` | Contains `GenerationContext`, `TypedDict` bundle types, and shared helpers. |
| `__init__.py` | Entry point. **Must be updated** when adding nodes via import from modules. |
| `docs/codemaps/structure.md` | Overview of the modular organization. |

## Common Pitfalls

| Don't | Do Instead |
|-------|-----------|
| Add separate image input/output to post-process nodes | Read/write `pipeline.image` from/to `UME_PIPELINE` |
| Create `GenerationContext` in a loader | Only `BlockSampler` creates `GenerationContext` |
| Return `MODEL`, `CLIP`, `VAE` separately from loaders | Return a single `UME_BUNDLE` dict |
| Forget `__init__.py` | Double-check registration after creating a new node class |
| Take native types as input without interop | Use `Pack Models Bundle` to convert native → UME, or `Unpack *` for UME → native |

## 🚨 Mandatory Verification Checklist

**Before marking any task as complete, you MUST verify:**

1. [ ] **`__init__.py` Updated**: Did you add the new node class to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`?
2. [ ] **Web Directory**: If the node has frontend code, is it in `web/` and registered?
3. [ ] **Syntax Check**: Did you do a final syntax check on the files you edited (especially big lists like mappings)?
4. [ ] **User Notification**: Did you tell the user *exactly* where to find the new node (Category/Name)?
