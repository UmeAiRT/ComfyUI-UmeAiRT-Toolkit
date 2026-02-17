# Project Structure Map

## High-Level Anatomy

| Directory/File | Description |
|----------------|-------------|
| `nodes.py` | **CORE**. Contains `UME_SHARED_STATE`, constants, and all Node Classes. |
| `__init__.py` | **ENTRY POINT**. Registers nodes with ComfyUI. Maps classes to names. |
| `web/` | Javascript files for UI extensions (e.g., color pickers, listeners). |
| `docs/` | Documentation (Architecture, Code Maps). |
| `.cursorrules` | AI Agent instructions. |
| `AGENTS.md` | Developer guide for AI Agents. |

## Sub-Modules (Cores)

Complex logic is offloaded to specialized core folders to keep `nodes.py` clean.

- `usdu_core/`: Logic for **Ultimate SD Upscale** integration.
- `facedetailer_core/`: Logic for **Face Detailer** integration.
- `image_saver_core/`: Logic for saving images.

## File Map: `nodes.py`

| Section | Content |
|---------|---------|
| `imports` | Standard imports & ComfyUI internal imports (`folder_paths`, `samplers`). |
| `Global Storage` | `UME_SHARED_STATE = {}` Definition. |
| `Internal Keys` | Constants defining dictionary keys. |
| `Variables Nodes` | Input/Output nodes for simple types (Float, Int, String). |
| `Loaders` | `WirelessImageLoader` & `WirelessCheckpointLoader`. |
| `Logic Nodes` | `WirelessKSampler`, `WirelessUltimateUpscale`, etc. |

## File Map: `__init__.py`

| Section | Content |
|---------|---------|
| `Imports` | Imports all node classes from `nodes.py`. |
| `NODE_CLASS_MAPPINGS` | Dictionary mapping `ClassName` -> `Class`. |
| `NODE_DISPLAY_NAME_MAPPINGS` | Dictionary mapping `ClassName` -> `"Human Readable Name"`. |
| `WEB_DIRECTORY` | Points ComfyUI to the `web/` folder. |
