# ⬡ Source Image Output

> Pass-through node for source image and mask — used in inpaint/composite workflows.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `source_image` | `IMAGE` | ✅ | The original source image |
| `source_mask` | `MASK` | ❌ | Optional mask (white = areas to modify) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `source_image` | `IMAGE` | Pass-through source image |
| `source_mask` | `MASK` | Mask (zeros if not provided) |

<!-- TODO: Screenshot — Source Image Output in an inpaint workflow -->
<!-- PLACEHOLDER: Show Source Image Output connected to Inpaint Composite alongside a KSampler output -->
