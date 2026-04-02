# UmeAiRT Toolkit - Agent Development Guide

> Instructions for AI coding agents working on this project.
> For architecture details, see `/docs/codemaps/`.
>
> This file follows the [AGENTS.md](https://agents.md) standard.

## Project Overview

ComfyUI Custom Nodes toolkit organized into 8 menu categories:
1. **Loaders**: Model loading nodes (Checkpoint, FLUX, Z-IMG, Bundle) + LoRA stacks.
2. **Inputs**: Generation Settings and Prompt editors.
3. **Image**: Image loading, processing (Img2Img/Inpaint/Outpaint), and ControlNet.
4. **Sampler**: Central hub (`BlockSampler`) using hub-and-spoke architecture with typed objects (`UME_BUNDLE`, `UME_SETTINGS`, `UME_PIPELINE`).
5. **Post-Process**: Upscalers and Detailers operating on `GenerationContext` (UME_PIPELINE).
6. **Output**: Image saving.
7. **Interop**: Pack/Unpack nodes for native ComfyUI compatibility.
8. **Utils**: Standalone nodes (Signature, Bundle Downloader).

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
                                           ├──▶ Post-Process nodes (read/write gen_pipe.image)
                                           └──▶ ImageSaver (reads gen_pipe.image)
```

**Key types:**

| Type | Content | Produced by |
|------|---------|-------------|
| `UME_BUNDLE` | `{model, clip, vae, model_name}` | All Loader nodes |
| `UME_SETTINGS` | `{width, height, steps, cfg, sampler_name, scheduler, seed}` | GenerationSettings |
| `UME_PIPELINE` | `GenerationContext` object (image + all context) | BlockSampler |
| `UME_IMAGE` | `{image, mask, mode, denoise, outpaint_target_w, outpaint_target_h, outpaint_align}` | BlockImageLoader → BlockImageProcess |

**Rules:**
- Post-process nodes receive `UME_PIPELINE`, read `gen_pipe.image`, process, update `gen_pipe.image`, return `UME_PIPELINE`.
- Never create `GenerationContext` outside the `BlockSampler`.
- All pipeline/generation parameters are named `gen_pipe`.

### Manifest & Auto-Download Architecture
- The toolkit relies on a local and remote architecture (`umeairt_bundles.json` and a remote HuggingFace `model_manifest.json`) to map models (Upscale, BBox, SegM) to their download URLs.
- **Dynamic UI**: Nodes like `⬡ UltimateSD Upscale` merge local scanned models with remote manifest entries. Remote files are prefixed with `[⬇️]` in dropdowns and are auto-downloaded on execution via `download_bundle_files` before processing.
- **Outpaint Logic**: The passive `⬡ Image Process (Outpaint)` node sets target dimensions in `UME_IMAGE`. The `BlockSampler` uses this to apply **Reflect Padding + Moderate Gaussian Blur** prior to VAE encoding. This preserves local textures at the boundaries and totally eliminates "box" or "barcode" stretching artifacts!



### Coding Standards

**Naming:**

- Class Names: `UmeAiRT_` prefix (e.g., `UmeAiRT_BlockSampler`).
- Display Names: Prefixed with `⬡` for visual identification (e.g., `⬡ KSampler`, `⬡ Checkpoint Loader`).
- Output names: `model_bundle` for loaders, `gen_pipe` for sampler/post-process.
- Pipeline parameters: Always use `gen_pipe` (not `pipeline` or `generation`).

**Registration:**

- All new nodes **MUST** be registered in `__init__.py` in two places:
    1. `NODE_CLASS_MAPPINGS`
    2. `NODE_DISPLAY_NAME_MAPPINGS`

### Advanced Inputs (Vue 2.0 Engine)

ComfyUI's Nodes 2.0 (Vue) layout engine natively suffers from a "ghost padding" bug when using `"advanced": True`. Visually hidden inputs still mathematically reserve their height in the node's `min-height`, causing massive empty gaps.
**The UmeAiRT Solution**: The `web/umeairt_colors.js` script contains an automated `computeSize` interceptor that subtracts the mathematical height of advanced widgets for all nodes prefixed with `UmeAiRT_`. 
**Rule**: You are free to use `"advanced": True` directly in your Python `INPUT_TYPES` for optional or heavy sliders. You do **not** need to split nodes into `_Simple` and `_Advanced` variants just to fix spacing gaps, as the Javascript layer will dynamically handle the CSS flex constraints for any `UmeAiRT_` node.

### File Structure

- `modules/common.py`: `GenerationContext` class, `TypedDict` bundle types (`UmeBundle`, `UmeSettings`, `UmeImage`), pipeline helpers (`extract_pipeline_params`, `validate_bundle`, `PipelineParams`), shared helpers (`resize_tensor`, `encode_prompts`, `apply_outpaint_padding`), `KNOWN_DIT_MODELS` constant.
- `modules/logger.py`: Standard logging utility.
- `modules/optimization_utils.py`: Environment and optimization checks.
- `modules/extra_samplers.py`: Custom KSampler algorithms (SA-Solver, RES Multistep).
- `modules/block_nodes.py`: Re-export shim for backward compatibility — imports from sub-modules.
- `modules/block_inputs.py`: LoRA blocks, ControlNet, GenerationSettings (→ `UME_SETTINGS`), Image Loader/Process, Prompt Inputs.
- `modules/block_loaders.py`: Model Loaders (→ `UME_BUNDLE`), BundleAutoLoader, shared download helpers.
- `modules/block_sampler.py`: BlockSampler hub (→ `UME_PIPELINE`).
- `modules/logic_nodes.py`: Pipeline-aware Upscalers, Detailers, and Detail Daemon.
- `modules/image_nodes.py`: Image loading, processing, saving (pipeline-aware).
- `modules/model_nodes.py`: Multi-LoRA Loader.
- `modules/utils_nodes.py`: Labels, debuggers, Pack/Unpack interoperability nodes, Bundle Downloader.
- `__init__.py`: Registration and exposing nodes to ComfyUI.
- `web/`: Javascript extensions (UI tweaks, colors, Nodes 2.0 enforcements).
- `*/core/`: Integrated libraries (e.g., `usdu_core`, `seedvr2_core`).
- `vendor/comfyui_gguf/`: Vendored implementation of `ComfyUI-GGUF` for `.gguf` weight loading.
- `tests/`: Unit tests (run with `python tests/test_*.py -v`).
- `.github/workflows/ci.yml`: GitHub Actions CI (Python 3.10-3.12, CPU PyTorch).

## UI & Styling (Node Colors)

Nodes are color-coded by category in `web/umeairt_colors.js`:

| Category | Color Family | Hex (Bg/Fg) | Menu Location |
|----------|--------------|-------------|---------------|
| **Settings / Controls**   | Amber / Bronze | `#4A290B` / `#935116` | `UmeAiRT/Inputs`, `UmeAiRT/Image` |
| **Model / Files**         | Deep Blue      | `#0A2130` / `#154360` | `UmeAiRT/Loaders` |
| **Prompts**               | Dark Green     | `#0A2D19` / `#145A32` | `UmeAiRT/Inputs` |
| **LoRA**                  | Violet         | `#25122D` / `#4A235A` | `UmeAiRT/Loaders/LoRA` |
| **Samplers (Processors)** | Slate Gray     | `#1A252F` / `#2C3E50` | `UmeAiRT/Sampler` |
| **Post-Processing**       | Pale Blue / Teal | `#123851` / `#2471A3` | `UmeAiRT/Post-Process` |
| **Utilities**             | Dark Gray      | `#1A252F` / `#34495E` | `UmeAiRT/Utils` |
| **Image Inputs**          | Rust Red       | `#35160D` / `#6B2D1A` | `UmeAiRT/Image` |

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

1. **Dependency Synchronization**: Always update `pyproject.toml` instantly when adding a new package to `requirements.txt`. They must mirror each other to guarantee seamless node installation for users. The `test_registration` suite validates this.
2. **Proper Exception Handling**: **NEVER** use bare exceptions (`except:` or `except: pass`). Always catch specific exceptions or use `except Exception as e:` and log the error via `log_node()` so failures are visible during debugging.
3. **Changelog Maintenance**: All notable modifications, bug fixes, or additions must be immediately documented in `CHANGELOG.md` following the *Keep a Changelog* format.
4. **Tooltip Requirement**: Every `INPUT_TYPES` parameter **MUST** have a `"tooltip"` key with a beginner-friendly description. The `test_tooltips` suite enforces this.
5. **Test Coverage**: Run `python tests/test_*.py -v` before submitting changes. CI runs automatically on push.

## Critical Files

| File | Notes |
|------|-------|
| `modules/common.py` | Contains `GenerationContext`, `TypedDict` bundle types, pipeline helpers, and shared utilities. |
| `__init__.py` | Entry point. **Must be updated** when adding nodes via import from modules. |
| `docs/codemaps/structure.md` | Overview of the modular organization. |
| `TODO.md` | Technical backlog (remaining items from critical analysis). |
| `tests/test_registration.py` | Validates NODE_CLASS_MAPPINGS ↔ NODE_DISPLAY_NAME_MAPPINGS sync, dep sync. |
| `tests/test_tooltips.py` | Regression test: every input must have a tooltip. |

## Common Pitfalls

| Don't | Do Instead |
|-------|-----------|
| Add separate image input/output to post-process nodes | Read/write `gen_pipe.image` from/to `UME_PIPELINE` |
| Create `GenerationContext` in a loader | Only `BlockSampler` creates `GenerationContext` |
| Return `MODEL`, `CLIP`, `VAE` separately from loaders | Return a single `UME_BUNDLE` dict |
| Forget `__init__.py` | Double-check registration after creating a new node class |
| Take native types as input without interop | Use `Pack Models Bundle` to convert native → UME, or `Unpack *` for UME → native |
| Name pipeline param `pipeline` or `generation` | Use `gen_pipe` everywhere |
| Add input without tooltip | Add `"tooltip": "description"` to every input dict |
| Skip tests | Run `python tests/test_*.py -v` before committing |
| Duplicate pipeline extraction | Use `extract_pipeline_params()` from `common.py` |
| Duplicate KNOWN_DIT_MODELS | Import from `common.py` |

## 🚨 Mandatory Verification Checklist

**Before marking any task as complete, you MUST verify:**

1. [ ] **`__init__.py` Updated**: Did you add the new node class to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`?
2. [ ] **Web Directory**: If the node has frontend code, is it in `web/` and registered?
3. [ ] **Tooltips**: Does every input have a beginner-friendly `"tooltip"` key?
4. [ ] **Tests Pass**: Did you run `python tests/test_*.py -v` and all tests pass?
5. [ ] **Syntax Check**: Did you do a final syntax check on the files you edited (especially big lists like mappings)?
6. [ ] **User Notification**: Did you tell the user *exactly* where to find the new node (Category/Name)?
