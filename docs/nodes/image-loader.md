# ⬡ Image Loader

> Load a source image from disk for img2img, inpaint, or outpaint workflows.

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image` | `COMBO` | ✅ | Select an image file from ComfyUI's `input/` directory (supports upload) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Bundle containing image + mask + default settings (mode=img2img, denoise=0.75) |
| `image` | `IMAGE` | Raw image tensor (for standard ComfyUI nodes) |
| `mask` | `MASK` | Raw mask tensor |

!!! tip "Pair with Image Process"
    The Image Loader outputs a default `img2img` bundle. Use [Image Process](image-process.md) to set the mode (inpaint, outpaint), denoise, auto-resize, and padding.
