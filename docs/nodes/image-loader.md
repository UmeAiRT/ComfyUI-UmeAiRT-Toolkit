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

!!! note "Bundle-only output"
    The Image Loader returns only the `UME_IMAGE` bundle. To access the raw `IMAGE` or `MASK` tensors (e.g., for native ComfyUI nodes), use **⬡ Unpack Image Bundle**.

!!! tip "Pair with Image Process"
    The Image Loader outputs a default `img2img` bundle. Use [Image Process](image-process.md) to set the mode (inpaint, outpaint), denoise, or auto-resize.
