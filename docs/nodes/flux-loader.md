# ⬡ FLUX Loader

> Load a FLUX architecture model with dual text encoders (CLIP-L + T5-XXL) and VAE.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `diff_model` | `COMBO` | ✅ | FLUX diffusion model (e.g. `flux1-dev-fp8.safetensors` or GGUF variant) |
| `clip_1` | `COMBO` | ✅ | First text encoder — typically CLIP-L (`clip_l.safetensors`) |
| `clip_2` | `COMBO` | ✅ | Second text encoder — typically T5-XXL (`t5xxl_fp16.safetensors` or GGUF) |
| `vae` | `COMBO` | ✅ | VAE model (e.g. `ae.safetensors`) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model` | `MODEL` | FLUX diffusion model |
| `clip` | `CLIP` | Dual text encoder (CLIP-L + T5-XXL) |
| `vae` | `VAE` | Autoencoder |
| `model_name` | `STRING` | Diffusion model filename |

## GGUF Support

The FLUX Loader automatically detects `.gguf` files and routes them through the built-in GGUF loader. You can mix formats:

- **Model**: `.safetensors` or `.gguf` (Q4_K_M, Q8_0, etc.)
- **CLIP**: `.safetensors` or `.gguf` (individually — you can use GGUF T5 with safetensors CLIP-L)

!!! tip "VRAM Recommendations"
    | Model Format | VRAM Required |
    |-------------|---------------|
    | FLUX Dev FP16 | ~24 GB |
    | FLUX Dev FP8 (e4m3fn) | ~12 GB |
    | FLUX Dev GGUF Q4_K_M | ~6 GB |

<!-- TODO: Screenshot — FLUX Loader node with model/clip/vae dropdowns visible -->
<!-- PLACEHOLDER: Show the FLUX Loader with typical file selections (flux1-dev, clip_l, t5xxl, ae) -->
