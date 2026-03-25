# ⬡ UltimateSD Upscale

> Tiled upscaling with redraw — upscales the generated image by processing it in overlapping tiles.

## Simple Version

Connects to a `UME_PIPELINE` and upscales inplace.

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | Pipeline from KSampler with generated image |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with upscaled image |

## Advanced Version

Full control over upscale parameters (scale factor, tile size, overlap, etc.).

<!-- TODO: Screenshot — UltimateSD Upscale (Advanced) node showing upscale parameters -->
<!-- PLACEHOLDER: Show the Advanced variant with visible parameters: upscale_by=2.0, tile_width=512, tile_height=512, and a before/after comparison -->

!!! tip "When to use"
    Use UltimateSD Upscale for **traditional tiled upscaling** with model redraw. For AI-native upscaling with better coherence, try [SeedVR2 Upscale](seedvr2-upscale.md).
