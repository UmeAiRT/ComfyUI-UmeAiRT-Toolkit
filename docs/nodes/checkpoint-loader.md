# ⬡ Checkpoint Loader

> Load a standard SD 1.5 / SDXL checkpoint with optional external VAE.

## Simple Version

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `ckpt_name` | `COMBO` | ✅ | — | Select a checkpoint file from `models/checkpoints/` |
| `vae_name` | `COMBO` | ✅ | Baked VAE | External VAE, or use the one embedded in the checkpoint |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `model` | `MODEL` | The loaded diffusion model |
| `clip` | `CLIP` | Text encoder for prompt processing |
| `vae` | `VAE` | Variational Autoencoder for image encoding/decoding |
| `model_name` | `STRING` | Checkpoint filename for downstream display |

!!! tip "When to use"
    Use this node when you want **direct access** to MODEL/CLIP/VAE outputs for wiring to standard ComfyUI nodes. For a fully bundled approach, use the Advanced version below.

<!-- TODO: Screenshot — Checkpoint Loader (Simple) connected to standard ComfyUI nodes -->
<!-- PLACEHOLDER: Show the simple loader with its 4 outputs wired to CLIPTextEncode, KSampler, and VAEDecode -->

---

## Advanced Version {#advanced}

> All-in-one loader: checkpoint + VAE + LoRA stack + prompts + generation settings → single `UME_BUNDLE` output.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `ckpt_name` | `COMBO` | ✅ | — | Checkpoint file |
| `vae_name` | `COMBO` | ✅ | Baked VAE | External VAE override |
| `clip_skip` | `INT` | ✅ | -1 | CLIP layers to skip (-1 = none, -2 = common for anime) |
| `positive` | `STRING` | ✅ | — | Positive prompt (what you want) |
| `negative` | `STRING` | ✅ | — | Negative prompt (what to avoid) |
| `width` | `INT` | ✅ | 512 | Output width (64–8192, step 8) |
| `height` | `INT` | ✅ | 768 | Output height (64–8192, step 8) |
| `batch_size` | `INT` | ✅ | 1 | Images per batch (1–64) |
| `steps` | `INT` | ✅ | 20 | Sampling steps (1–200) |
| `cfg` | `FLOAT` | ✅ | 7.0 | CFG scale (0–50) |
| `sampler_name` | `COMBO` | ✅ | — | Sampling algorithm |
| `scheduler` | `COMBO` | ✅ | — | Noise schedule |
| `seed` | `INT` | ✅ | 0 | Random seed for reproducibility |
| `denoise` | `FLOAT` | ✅ | 1.0 | Denoising strength (1.0 = full, <1.0 = img2img) |
| `lora_stack` | `LORA_STACK` | ❌ | — | Optional LoRA stack from a LoRA Block node |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `model_bundle` | `UME_BUNDLE` | Complete bundle ready for KSampler |

!!! note "Self-contained node"
    The Advanced loader is a self-contained workflow-in-a-node. It loads the model, applies LoRAs, encodes prompts, and packs everything into a single bundle. Ideal for quick setups.

<!-- TODO: Screenshot — Checkpoint Loader (Advanced) connected to KSampler → Image Saver -->
<!-- PLACEHOLDER: Show the advanced loader as a compact 3-node workflow (Loader → Sampler → Saver) -->
