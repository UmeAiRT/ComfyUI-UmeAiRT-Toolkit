# ⬡ SeedVR2 Upscale

> AI-native upscaling using the SeedVR2 tiling and stitching engine.

## Simple Version

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | Pipeline from KSampler |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with upscaled image |

## Advanced Version

Additional control over the upscaling process.

!!! warning "VRAM Requirements"
    SeedVR2 requires approximately **6 GB of free VRAM**. The node includes automatic VRAM management — it will unload cached models if necessary to free space.

<!-- TODO: Screenshot — SeedVR2 Upscale in a workflow: KSampler → SeedVR2 → Image Saver -->
<!-- PLACEHOLDER: Show a SeedVR2 node between KSampler and Image Saver, with the console showing "SeedVR2 VRAM Check: OK" -->
