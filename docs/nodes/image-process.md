# ⬡ Image Process

> Pre-process an image bundle: set mode, denoise, resize, outpaint padding, and mask blur.

## Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `image_bundle` | `UME_IMAGE` | ✅ | — | Input image bundle from Image Loader |
| `denoise` | `FLOAT` | ✅ | 0.75 | How much the AI changes (1.0 = full redraw, 0.5 = keep half, slider) |
| `mode` | `COMBO` | ✅ | img2img | Processing mode: `img2img`, `inpaint`, `outpaint`, `txt2img` |
| `auto_resize` | `BOOLEAN` | ❌ | OFF | Resize image to match Generation Settings dimensions |
| `mask_blur` | `INT` | ❌ | 10 | Soften mask edges for inpaint/outpaint blending |
| `padding_left` | `INT` | ❌ | 0 | Outpaint pixels — left side |
| `padding_top` | `INT` | ❌ | 0 | Outpaint pixels — top |
| `padding_right` | `INT` | ❌ | 0 | Outpaint pixels — right side |
| `padding_bottom` | `INT` | ❌ | 0 | Outpaint pixels — bottom |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Processed image bundle ready for KSampler or ControlNet |

## Modes

| Mode | Behavior |
|------|----------|
| `img2img` | Transform the image. Mask is ignored. |
| `inpaint` | Fill masked areas (white = modify). Mask blur applied. |
| `outpaint` | Extend image edges using padding values. |
| `txt2img` | Forces denoise=1.0, ignores mask. Image used only for dimensions. |

<!-- TODO: Screenshot — Image Process node showing outpaint mode with padding visible -->
<!-- PLACEHOLDER: Show Image Loader → Image Process (mode=outpaint, padding_right=256) → KSampler chain -->
