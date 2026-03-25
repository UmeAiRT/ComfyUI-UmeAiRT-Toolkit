# ⬡ Image Loader

> Load a source image from disk for img2img, inpaint, or outpaint workflows.

## Simple Version

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image` | `COMBO` | ✅ | Select an image file from ComfyUI's `input/` directory (supports upload) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Bundle containing image + mask + default settings (mode=img2img, denoise=0.75) |

## Advanced Version

Same inputs as the simple version, but with additional raw outputs:

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image_bundle` | `UME_IMAGE` | Complete image bundle |
| `image` | `IMAGE` | Raw image tensor (for standard ComfyUI nodes) |
| `mask` | `MASK` | Raw mask tensor |

!!! tip "Pair with Image Process"
    The Image Loader outputs a default `img2img` bundle. Use [Image Process](image-process.md) to set the mode (inpaint, outpaint), denoise, auto-resize, and padding.

<!-- TODO: Screenshot — Image Loader (Advanced) with its 3 outputs visible -->
<!-- PLACEHOLDER: Show the advanced loader with an image selected and the 3 outputs wired to different nodes -->
