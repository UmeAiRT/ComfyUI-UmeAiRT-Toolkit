# ⬡ KSampler

> Central hub node: receives models, settings, prompts, and images — samples and produces the generation pipeline.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model_bundle` | `UME_BUNDLE` | ✅ | Model + CLIP + VAE from a Loader node |
| `settings` | `UME_SETTINGS` | ✅ | Parameters from Generation Settings |
| `positive` | `POSITIVE` | ✅ | Positive prompt text |
| `negative` | `NEGATIVE` | ❌ | Negative prompt text |
| `loras` | `UME_LORA_STACK` | ❌ | LoRA stack from LoRA Block nodes |
| `image` | `UME_IMAGE` | ❌ | Image bundle for img2img/inpaint/outpaint |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Complete pipeline with generated image, ready for post-processing or saving |

## What Happens Inside

The KSampler orchestrates the full generation pipeline:

1. **Validates** the model bundle (model, clip, vae must all be present)
2. **Applies LoRAs** from the stack (if connected)
3. **Encodes prompts** via CLIP (with caching for repeated prompts)
4. **Applies ControlNets** from image bundle (if present)
5. **Prepares latent** — empty for txt2img, or VAE-encoded for img2img/inpaint
6. **Samples** using the configured sampler + scheduler + steps
7. **Decodes** the latent to pixel space via VAE
8. **Packs** everything into a `UME_PIPELINE` for downstream nodes

!!! tip "Optimization"
    The KSampler caches prompt encodings and ControlNet models. If you change only the seed, re-encoding is skipped for faster iteration.

<!-- TODO: Screenshot — KSampler with all inputs connected (model + settings + prompt + LoRA + image) -->
<!-- PLACEHOLDER: Show a full KSampler node with 5 connected inputs and the gen_pipe output going to Image Saver -->
