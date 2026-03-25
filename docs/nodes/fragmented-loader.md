# ⬡ Fragmented Loader

> Load models distributed as folder-based architectures (HuggingFace Diffusers format).

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model_path` | `COMBO` | ✅ | Select a model folder from `models/diffusion_models/`. Folders contain `index.json` + sharded safetensors |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model` | `MODEL` | Loaded diffusion model |
| `clip` | `CLIP` | Text encoder |
| `vae` | `VAE` | Autoencoder |
| `model_name` | `STRING` | Folder name |

!!! note "Folder structure"
    The Fragmented Loader scans subdirectories of `diffusion_models/` for model folders. A valid folder typically contains `model_index.json` and multiple sharded `.safetensors` files.

<!-- TODO: Screenshot — Fragmented Loader with model folder dropdown -->
<!-- PLACEHOLDER: Show the dropdown listing available model folders -->
