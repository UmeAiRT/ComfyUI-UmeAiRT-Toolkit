# ⬡ SeedVR2 Upscale

> AI-native upscaling using the SeedVR2 tiling and stitching engine. Tile/blending parameters are available via **Show advanced inputs**.

### Inputs

| Name | Type | Required | Advanced | Default | Description |
|------|------|----------|----------|---------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | | — | Pipeline from KSampler |
| `enabled` | `BOOLEAN` | ✅ | | ON | Toggle upscaling on/off |
| `model` | `COMBO` | ✅ | | seedvr2_ema_3b_fp8 | SeedVR2 model selection |
| `upscale_by` | `FLOAT` | ✅ | | 2.0 | Scale factor (1–8x) |
| `tile_width` | `INT` | ✅ | ✅ | 512 | Tile width in pixels |
| `tile_height` | `INT` | ✅ | ✅ | 512 | Tile height in pixels |
| `mask_blur` | `INT` | ✅ | ✅ | 0 | Tile edge softening |
| `tile_padding` | `INT` | ✅ | ✅ | 32 | Tile overlap in pixels |
| `tile_upscale_resolution` | `INT` | ✅ | ✅ | 1024 | Per-tile processing resolution |
| `tiling_strategy` | `COMBO` | ✅ | ✅ | Chess | Tile layout strategy |
| `anti_aliasing_strength` | `FLOAT` | ✅ | ✅ | 0.0 | Edge smoothing between tiles |
| `blending_method` | `COMBO` | ✅ | ✅ | auto | Tile merging algorithm |
| `color_correction` | `COMBO` | ✅ | ✅ | lab | Color matching between tiles |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with upscaled image |

!!! warning "VRAM Requirements"
    SeedVR2 requires approximately **6 GB of free VRAM**. The node includes automatic VRAM management — it will unload cached models if necessary to free space.
