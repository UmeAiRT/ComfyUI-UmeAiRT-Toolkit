# Node Reference

All UmeAiRT nodes use the **⬡** prefix in ComfyUI and are organized into categories in the right-click menu under `UmeAiRT/`.

## Loaders

Load models, VAEs, and text encoders into memory.

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Checkpoint Loader](checkpoint-loader.md) | ⬡ Checkpoint Loader | MODEL, CLIP, VAE | SD 1.5 / SDXL single-file checkpoints |
| [Checkpoint Loader (Advanced)](checkpoint-loader.md#advanced) | ⬡ Checkpoint Loader (Advanced) | UME_BUNDLE | All-in-one: model + prompts + settings |
| [FLUX Loader](flux-loader.md) | ⬡ FLUX Loader | MODEL, CLIP, VAE | FLUX architecture (dual CLIP) |
| [Z-IMG Loader](zimg-loader.md) | ⬡ Z-IMG Loader | MODEL, CLIP, VAE | Z-IMAGE / Lumina2 architecture |
| [Bundle Auto-Loader](bundle-loader.md) | ⬡ 📦 Bundle Auto-Loader | UME_BUNDLE | Auto-download + load from manifest |

## Generation

Configure generation parameters and prepare images for processing.

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Generation Settings](generation-settings.md) | ⬡ Generation Settings | UME_SETTINGS | Dimensions, steps, CFG, sampler, seed |
| [Image Loader](image-loader.md) | ⬡ Image Loader | UME_IMAGE | Load source image for img2img/inpaint |
| [Image Process](image-process.md) | ⬡ Image Process | UME_IMAGE | All-in-one: mode, denoise, resize, outpaint |
| [Image Process (Img2Img)](image-process.md#-image-process-img2img) | ⬡ Image Process (Img2Img) | UME_IMAGE | Dedicated img2img |
| [Image Process (Inpaint)](image-process.md#-image-process-inpaint) | ⬡ Image Process (Inpaint) | UME_IMAGE | Dedicated inpainting |
| [Image Process (Outpaint)](image-process.md#-image-process-outpaint) | ⬡ Image Process (Outpaint) | UME_IMAGE | Target dimensions + alignment |
| [LoRA Blocks](lora-blocks.md) | ⬡ LoRA 1x/3x/5x/10x | UME_LORA_STACK | Stack LoRA models with strengths |
| [ControlNet](controlnet.md) | ⬡ ControlNet Apply/Process | UME_IMAGE | Apply ControlNet guidance |
| [Prompt Inputs](prompt-inputs.md) | ⬡ Positive/Negative Prompt | POSITIVE/NEGATIVE | Text prompt editors |

## Sampling & Post-Process

Generate and enhance images.

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [KSampler](ksampler.md) | ⬡ KSampler | UME_PIPELINE | Central sampling hub |
| [UltimateSD Upscale](ultimate-upscale.md) | ⬡ UltimateSD Upscale | UME_PIPELINE | Tiled upscaling with redraw |
| [SeedVR2 Upscale](seedvr2-upscale.md) | ⬡ SeedVR2 Upscale | UME_PIPELINE | AI-native upscaling |
| [FaceDetailer](facedetailer.md) | ⬡ FaceDetailer | UME_PIPELINE | Automatic face enhancement |
| [Detailer Daemon](detailer-daemon.md) | ⬡ Detailer Daemon | UME_PIPELINE | Configurable face detection params |

## Image Output

Save and composite final images.

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Image Saver](image-saver.md) | ⬡ Image Saver | — | Save with metadata + naming |

## Pack / Unpack

Convert between UmeAiRT bundles and standard ComfyUI types.

See [Pack / Unpack Overview](pack-unpack.md) for the full list of 13 interoperability nodes.

## Utilities

Workflow helpers and diagnostics.

| Node | Display Name | Output | Use Case |
|------|-------------|--------|----------|
| [Bundle Downloader](bundle-downloader.md) | ⬡ 💾 Bundle Model Downloader | STRING | Pre-download models without loading |
