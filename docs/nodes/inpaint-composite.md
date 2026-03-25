# ⬡ Inpaint Composite

> Blend the AI-generated image back onto the source image using a mask.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `generated_image` | `IMAGE` | ✅ | The AI-generated result |
| `source_image` | `IMAGE` | ✅ | The original image to blend onto |
| `source_mask` | `MASK` | ❌ | Black/white mask: black = keep source, white = use generated |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `composite_image` | `IMAGE` | Blended result |

!!! note "Auto-resize"
    If the source image dimensions don't match the generated image, the source is automatically resized to match.

<!-- TODO: Screenshot — Inpaint composite workflow -->
<!-- PLACEHOLDER: Show a 3-way connection: Source Image Output → Inpaint Composite ← KSampler output, with the composited result preview -->
