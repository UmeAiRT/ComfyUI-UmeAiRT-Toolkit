# ⬡ ControlNet

Three nodes for applying ControlNet guidance to image bundles.

## ControlNet Apply (Simple)

Minimal ControlNet application with just model and strength.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `image_bundle` | `UME_IMAGE` | ✅ | — | Image bundle to apply ControlNet to |
| `control_net_name` | `COMBO` | ✅ | — | ControlNet model from `models/controlnet/` |
| `strength` | `FLOAT` | ✅ | 1.0 | Guidance strength (0–2, slider) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Bundle with ControlNet attached |

---

## ControlNet Apply (Advanced)

Full control over ControlNet timing and optional separate control image.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `image_bundle` | `UME_IMAGE` | ✅ | — | Input image bundle |
| `control_net_name` | `COMBO` | ✅ | — | ControlNet model |
| `strength` | `FLOAT` | ✅ | 1.0 | Guidance strength (0–10) |
| `start_percent` | `FLOAT` | ✅ | 0.0 | When guidance starts (0.0 = beginning) |
| `end_percent` | `FLOAT` | ✅ | 1.0 | When guidance ends (1.0 = final step) |
| `optional_control_image` | `IMAGE` | ❌ | — | Override: use this image instead of bundle image |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Bundle with ControlNet attached |

---

## ControlNet Process

Combined image pre-processing + ControlNet application.

### Inputs

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `image_bundle` | `UME_IMAGE` | ✅ | — | Input image bundle |
| `denoise` | `FLOAT` | ✅ | 0.75 | Denoising strength |
| `mode` | `COMBO` | ✅ | img2img | Processing mode: `img2img` or `txt2img` |
| `control_net_name` | `COMBO` | ✅ | — | ControlNet model |
| `strength` | `FLOAT` | ✅ | 1.0 | Guidance strength (0–2) |
| `gen_pipe` | `UME_PIPELINE` | ❌ | — | Pipeline for resize dimensions |
| `resize` | `BOOLEAN` | ❌ | OFF | Auto-resize to match generation settings |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Processed bundle with ControlNet attached |

<!-- TODO: Screenshot — ControlNet Apply (Advanced) with start/end percent visible -->
<!-- PLACEHOLDER: Show an Image Loader → ControlNet Apply (Advanced) → KSampler chain with start_percent=0.2 and end_percent=0.8 -->
