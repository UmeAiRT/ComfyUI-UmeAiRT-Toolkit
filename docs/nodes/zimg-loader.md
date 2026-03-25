# ⬡ Z-IMG Loader

> Load a Z-IMAGE / Lumina2 architecture model with Qwen text encoder and VAE.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `diff_model` | `COMBO` | ✅ | Z-IMAGE diffusion model (e.g. `z-image-turbo-bf16.safetensors` or GGUF) |
| `clip` | `COMBO` | ✅ | Qwen text encoder (e.g. `qwen3-4b.safetensors` or GGUF) |
| `vae` | `COMBO` | ✅ | VAE model (e.g. `ae.safetensors`) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model` | `MODEL` | Z-IMAGE diffusion model |
| `clip` | `CLIP` | Qwen text encoder |
| `vae` | `VAE` | Autoencoder |
| `model_name` | `STRING` | Diffusion model filename |

!!! tip "Z-IMAGE Turbo"
    Z-IMAGE Turbo models can generate high-quality images in 4-8 steps with the `sgm_uniform` scheduler.

<!-- TODO: Screenshot — Z-IMG Loader node in a workflow -->
<!-- PLACEHOLDER: Show the Z-IMG Loader connected to KSampler with typical Z-IMAGE settings -->
