# UmeAiRT Toolkit - Agent Development Guide

> Instructions for AI coding agents working on this project.
> For architecture details, see `/docs/codemaps/`.
>
> This file follows the [AGENTS.md](https://agents.md) standard.

## Project Overview

ComfyUI Custom Nodes toolkit focusing on a **"Wireless" workflow experience**.
Key characteristic: **Shared Global State (`UME_SHARED_STATE`)**.
Nodes act as "Setters" (Input) or "Getters" (Output/Processor), decoupling complex wiring.

## Critical Conventions

### Wireless Architecture (Global State)

**Never** pass latent/model/clip connections manually between Wireless nodes.
Use the `UME_SHARED_STATE` dictionary in `modules/common.py` to store and retrieve data.

- **Input Nodes**: Write to `UME_SHARED_STATE`.
- **Output/Process Nodes**: Read from `UME_SHARED_STATE`.

**Example:**

```python
# Input Node (Setter)
def set_val(self, model):
    UME_SHARED_STATE[KEY_MODEL] = model
    return ()

# Output Node (Getter)
def get_val(self):
    val = UME_SHARED_STATE.get(KEY_MODEL)
    if val is None:
        raise ValueError("No Wireless Model set!")
    return (val,)
```

### Coding Standards

**Naming:**

- Class Names: `UmeAiRT_` prefix (e.g., `UmeAiRT_WirelessKSampler`).
- internal Keys: `KEY_` prefix (e.g., `KEY_MODEL`, `KEY_VAE`).
- Display Names: Clear, user-friendly (e.g., "KSampler (Wireless)").

**Registration:**

- All new nodes **MUST** be registered in `__init__.py` in two places:
    1. `NODE_CLASS_MAPPINGS`
    2. `NODE_DISPLAY_NAME_MAPPINGS`

### File Structure

- `modules/common.py`: Shared constants, global state (`UME_SHARED_STATE`), and core utilities.
- `modules/logger.py`: Standard logging utility.
- `modules/optimization_utils.py`: Environment and optimization checks.
- `modules/*.py`: Node definitions categorized logically (`settings_nodes`, `logic_nodes`, `image_nodes`, etc.).
- `__init__.py`: Registration and exposing nodes to ComfyUI.
- `web/`: Javascript extensions (UI tweaks and Nodes 2.0 enforcements).
- `*/core/`: Integrated libraries (e.g., `usdu_core`, `seedvr2_core`).
- `vendor/comfyui_gguf/`: Vendored implementation of `ComfyUI-GGUF` for `.gguf` weight loading.

## UI & Styling (Node Colors)

Nodes are color-coded by category in `web/umeairt_colors.js`:

| Category | Color Family | Hex (Bg/Fg) | Examples |
|----------|--------------|-------------|----------|
| **Settings / Controls**   | Amber / Bronze | `#4A290B` / `#935116` | Generation Settings, Image Process, ControlNet |
| **Model / Files**         | Deep Blue      | `#0A2130` / `#154360` | Checkpoint Loader, VAE, CLIP |
| **Prompts**               | Dark Green     | `#0A2D19` / `#145A32` | Wireless Prompts |
| **LoRA**                  | Violet         | `#25122D` / `#4A235A` | LoRA Stacks |
| **Samplers (Processors)** | Slate Gray     | `#1A252F` / `#2C3E50` | Wireless KSampler, Block Sampler |
| **Post-Processing**       | Pale Blue / Teal | `#123851` / `#2471A3` | Ultimate Upscale, Face Detailer, Inpaint Composite |
| **Utilities**             | Dark Gray      | `#1A252F` / `#34495E` | Debug, Label, UmeAiRT Signature |
| **Image Inputs**          | Rust Red       | `#35160D` / `#6B2D1A` | Image Loaders |

## Project Maintenance & Stability Rules

To avoid regressions and maintain a stable, production-ready codebase, adhere strictly to the following rules:

1. **Dependency Synchronization**: Always update `pyproject.toml` instantly when adding a new package to `requirements.txt`. They must mirror each other to guarantee seamless node installation for users.
2. **Proper Exception Handling**: **NEVER** use bare exceptions (`except:` or `except: pass`). Always catch specific exceptions or use `except Exception as e:` and log the error via `log_node()` so failures are visible during debugging.
3. **Changelog Maintenance**: All notable modifications, bug fixes, or additions must be immediately documented in `CHANGELOG.md` following the *Keep a Changelog* format.

## Critical Files

| File | Notes |
|------|-------|
| `modules/common.py` | Contains `UME_SHARED_STATE` and shared configurations. |
| `__init__.py` | Entry point. **Must be updated** when adding nodes via import from modules. |
| `docs/codemaps/structure.md` | Overview of the modular organization. |

## Common Pitfalls

| Don't | Do Instead |
|-------|-----------|
| Create standard ComfyUI nodes | Create "Wireless" nodes that interact with `UME_SHARED_STATE` |
| Hardcode internal keys | Use the defined constants in `modules/common.py` (e.g., `KEY_LATENT`) |
| Forget `__init__.py` | Double-check registration after creating a new node class |

## 🚨 Mandatory Verification Checklist

**Before marking any task as complete, you MUST verify:**

1. [ ] **`__init__.py` Updated**: Did you add the new node class to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `__init__.py`?
2. [ ] **Web Directory**: If the node has frontend code, is it in `web/` and registered?
3. [ ] **Syntax Check**: Did you do a final syntax check on the files you edited (especially big lists like mappings)?
4. [ ] **User Notification**: Did you tell the user *exactly* where to find the new node (Category/Name)?
