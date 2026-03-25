# ⬡ Multi-LoRA Loader

> Apply up to 3 LoRA models to a MODEL + CLIP pair using direct wiring (no block bundles).

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model` | `MODEL` | ✅ | Base diffusion model |
| `clip` | `CLIP` | ✅ | Text encoder |
| `lora_1` | `BOOLEAN` | ✅ | Enable LoRA slot 1 |
| `lora_1_name` | `COMBO` | ✅ | Select LoRA model for slot 1 |
| `lora_1_strength` | `FLOAT` | ✅ | Strength for slot 1 (0–2, slider) |
| `lora_2` | `BOOLEAN` | ✅ | Enable LoRA slot 2 |
| `lora_2_name` | `COMBO` | ✅ | Select LoRA model for slot 2 |
| `lora_2_strength` | `FLOAT` | ✅ | Strength for slot 2 |
| `lora_3` | `BOOLEAN` | ✅ | Enable LoRA slot 3 |
| `lora_3_name` | `COMBO` | ✅ | Select LoRA model for slot 3 |
| `lora_3_strength` | `FLOAT` | ✅ | Strength for slot 3 |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `model` | `MODEL` | Model with LoRAs applied |
| `clip` | `CLIP` | CLIP with LoRAs applied |

!!! tip "Wired vs Block LoRA"
    This node uses **direct MODEL/CLIP wiring** — compatible with standard ComfyUI nodes. For the block-based approach (feeding into KSampler via `UME_LORA_STACK`), use the [LoRA Block nodes](lora-blocks.md) instead.

<!-- TODO: Screenshot — Multi-LoRA Loader between a Checkpoint Loader and KSampler -->
<!-- PLACEHOLDER: Show Checkpoint Loader → Multi-LoRA Loader → standard ComfyUI KSampler with 2 LoRAs toggled ON -->
