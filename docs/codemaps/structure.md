# Project Structure Map

## High-Level Anatomy

| Directory/File | Description |
|----------------|-------------|
| `__init__.py` | **ENTRY POINT**. Registers nodes with ComfyUI and handles theme/settings injection. |
| `modules/` | **CORE LOGIC**. Refactored modular node implementations. |
| `web/` | Javascript files for UI extensions (styling and Nodes 2.0 enforcements). |
| `docs/` | Internal architectural documentation and code maps. |
| `.cursorrules` | AI Agent instructions and coding standards. |
| `AGENTS.md` | Developer guide for AI Agents. |

## Sub-Modules (`modules/`)

The toolkit is organized into functional modules to maintain scalability:

- `common.py`: Global constants and the `UME_SHARED_STATE` dictionary.
- `logger.py`: Standardized colorized logging utility.
- `optimization_utils.py`: Environment checks (SageAttention, Triton, etc.).
- `settings_nodes.py`: Wireless Variable Setters/Getters (Steps, CFG, Prompts).
- `model_nodes.py`: Wireless and Block-based Model/LoRA loaders.
- `logic_nodes.py`: The "Brains" - Wireless Samplers, Upscalers, and Detailers.
- `block_nodes.py`: The Block-based equivalents of samplers, loaders, and the Bundle Auto-Loader.
- `image_nodes.py`: Wireless image loading, processing, and saving.
- `utils_nodes.py`: Labels, state debuggers, and bundle Unpack nodes.

## Core Directories (Vendored/Integrated)

- `facedetailer_core/`: Logic for face detection and enhancement.
- `seedvr2_core/`: Ported tiling upscaler for high-VRAM efficiency.
- `usdu_core/`: Integrated Ultimate SD Upscale logic.
- `image_saver_core/`: Robust image saving with metadata.
- `vendor/comfyui_gguf/`: GGUF model loading support.
- `vendor/aria2/`: Bundled aria2c binary for accelerated model downloads.

## Registration Workflow

1. Node classes are defined in `modules/`.
2. `__init__.py` imports necessary classes.
3. `NODE_CLASS_MAPPINGS` links ComfyUI internal keys to Python classes.
4. `NODE_DISPLAY_NAME_MAPPINGS` provides user-friendly titles.
5. `WEB_DIRECTORY` exposes the `web/` folder for frontend styling.
