# ⬡ 💾 Bundle Model Downloader

> Pre-download model bundles to disk without loading them into memory.

## Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `category` | `COMBO` | ✅ | Model family to download (e.g. `FLUX/Dev`, `Z-IMAGE/Turbo`) |
| `version` | `COMBO` | ✅ | Quantization variant (e.g. `fp16`, `GGUF_Q4`) |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `status` | `STRING` | Download status report (files downloaded, skipped, errors) |

## Use Cases

- **Pre-download** models on cloud instances (RunPod, Vast.ai) before running workflows
- **Batch-download** entire model families overnight
- **Verify** all required files are present before generation

!!! tip "Difference from Bundle Auto-Loader"
    The **Downloader** only downloads files — it doesn't load them into VRAM. Use it for pre-staging. The **[Bundle Auto-Loader](bundle-loader.md)** downloads AND loads.

<!-- TODO: Screenshot — Bundle Downloader showing download status output -->
<!-- PLACEHOLDER: Show the node with "FLUX/Dev" + "GGUF_Q4" selected and the status output showing "📥 FLUX/Dev/GGUF_Q4: 3 downloaded | 1 already present" -->
