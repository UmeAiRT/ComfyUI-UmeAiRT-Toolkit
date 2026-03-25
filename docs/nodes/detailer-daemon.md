# ⬡ Detailer Daemon

> Configurable parameters for face detection and enhancement in FaceDetailer.

## Simple Version

Provides basic detection parameters.

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | Pipeline to configure |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with FaceDetailer configuration |

## Advanced Version

Full control over detection thresholds, enhancement strength, and crop parameters.

!!! tip "What's a Daemon?"
    The Detailer Daemon doesn't process images itself — it **configures** how FaceDetailer will detect and enhance faces. Think of it as a settings node specifically for face processing.

<!-- TODO: Screenshot — Detailer Daemon (Advanced) node showing detection threshold and crop parameters -->
<!-- PLACEHOLDER: Show a Detailer Daemon node connected between KSampler and FaceDetailer in a workflow -->
