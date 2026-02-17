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
Use the `UME_SHARED_STATE` dictionary in `nodes.py` to store and retrieve data.

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

- `nodes.py`: Core logic and node definitions.
- `__init__.py`: Registration and exposing nodes to ComfyUI.
- `web/`: Javascript extensions (UI tweaks).
- `*/core/`: Sub-modules for complex logic (e.g., `usdu_core`).

## UI & Styling (Node Colors)

Nodes are color-coded by category in `web/umeairt_colors.js`:

| Category | Color Family | Hex (Bg/Fg) | Examples |
|----------|--------------|-------------|----------|
| **Settings / Controls**   | Amber / Bronze | `#4A290B` / `#935116` | Generation Settings, Image Process, ControlNet |
| **Model / Files**         | Deep Blue      | `#0A2130` / `#154360` | Checkpoint Loader, VAE, CLIP |
| **Prompts**               | Dark Green     | `#0A2D19` / `#145A32` | Prompt Block, Wireless Prompts |
| **LoRA**                  | Violet         | `#25122D` / `#4A235A` | LoRA Stacks, Multi-LoRA |
| **Samplers (Processors)** | Slate Gray     | `#1A252F` / `#2C3E50` | Wireless KSampler, Block Sampler |
| **Upscale / Detailer**    | Pale Blue      | `#123851` / `#2471A3` | Ultimate Upscale, Face Detailer, Inpaint Composite |
| **Utilities**             | Dark Gray      | `#1A252F` / `#34495E` | Debug, Label |
| **Image Inputs**          | Rust Red       | `#35160D` / `#6B2D1A` | Image Loaders |

## Critical Files

| File | Notes |
|------|-------|
| `nodes.py` | Contains `UME_SHARED_STATE` and most node logic. |
| `__init__.py` | Entry point. **Must be updated** when adding nodes. |
| `docs/codemaps/wireless.md` | Detailed list of available Wireless keys. |

## Common Pitfalls

| Don't | Do Instead |
|-------|-----------|
| Create standard ComfyUI nodes | Create "Wireless" nodes that interact with `UME_SHARED_STATE` |
| Hardcode internal keys | Use the defined constants in `nodes.py` (e.g., `KEY_LATENT`) |
| Forget `__init__.py` | Double-check registration after creating a new node class |
