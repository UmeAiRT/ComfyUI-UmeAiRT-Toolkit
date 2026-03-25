# ⬡ FaceDetailer

> Automatically detect and enhance faces in generated images.

## Simple Version

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `gen_pipe` | `UME_PIPELINE` | ✅ | Pipeline with generated image |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `gen_pipe` | `UME_PIPELINE` | Pipeline with enhanced faces |

## Advanced Version

Full control over face detection and enhancement parameters.

Requires a **BBOX detector model** — use the [BBOX Detector Loader](#bbox-detector-loader) node.

## BBOX Detector Loader {#bbox-detector-loader}

Loads a face/body detection model for FaceDetailer.

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model_name` | `COMBO` | ✅ | Detection model from `models/bbox/` (e.g. `face_yolov8m.pt`) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `bbox_detector` | `BBOX_DETECTOR` | Loaded detector for FaceDetailer |

!!! tip "Recommended detector"
    Use `face_yolov8m.pt` for face detection. Place it in `ComfyUI/models/bbox/`.

<!-- TODO: Screenshot — FaceDetailer workflow: KSampler → FaceDetailer → Image Saver, with detected faces visible -->
<!-- PLACEHOLDER: Show a portrait generation where FaceDetailer has enhanced the face. Include a side-by-side before/after if possible. -->
