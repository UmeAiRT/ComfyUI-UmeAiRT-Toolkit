# ⬡ Detailer Daemon

> Configurable face detection and enhancement parameters for FaceDetailer. Advanced schedule parameters are available via **Show advanced inputs**.

### Inputs

| Name | Type | Required | Advanced | Default | Description |
|------|------|----------|----------|---------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | | — | Pipeline to configure |
| `schedule` | `COMBO` | ✅ | ✅ | — | Enhancement schedule |
| `denoise_start` | `FLOAT` | ✅ | ✅ | 1.0 | Start denoising value |
| `denoise_end` | `FLOAT` | ✅ | ✅ | 0.5 | End denoising value |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with Detailer Daemon configuration |

!!! tip "What's a Daemon?"
    The Detailer Daemon doesn't process images itself — it **configures** how FaceDetailer will detect and enhance faces. Think of it as a settings node specifically for face processing.
