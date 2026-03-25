# Pack / Unpack Nodes

> 13 interoperability nodes to convert between UmeAiRT bundles and standard ComfyUI types.

## Pack

| Node | Display Name | Inputs | Output | Description |
|------|-------------|--------|--------|-------------|
| Pack Bundle | ⬡ Pack Models Bundle | MODEL, CLIP, VAE | `UME_BUNDLE` | Wrap standard ComfyUI model outputs into a UmeAiRT bundle |

## Unpack

| Node | Display Name | Input | Outputs | Description |
|------|-------------|-------|---------|-------------|
| Unpack Pipeline | ⬡ Unpack Pipeline | `UME_PIPELINE` | IMAGE, MASK, MODEL, CLIP, VAE, ... | Extract all components from a pipeline |
| Unpack Models Bundle | ⬡ Unpack Models Bundle | `UME_BUNDLE` | MODEL, CLIP, VAE, STRING | Extract individual model components |
| Unpack Settings | ⬡ Unpack Settings | `UME_SETTINGS` | INT, INT, INT, FLOAT, STRING, STRING, INT | Extract width, height, steps, cfg, sampler, scheduler, seed |
| Unpack Image Bundle | ⬡ Unpack Image Bundle | `UME_IMAGE` | IMAGE, MASK | Extract raw tensors |
| Unpack Prompts Bundle | ⬡ Unpack Prompts Bundle | `UME_PIPELINE` | STRING, STRING | Extract positive/negative prompts |
| Unpack Settings Bundle | ⬡ Unpack Settings Bundle | `UME_PIPELINE` | INT, INT, INT, FLOAT, STRING, STRING, INT | Extract settings from pipeline |
| Unpack Faces | ⬡ Unpack Faces | `UME_PIPELINE` | Faces data | Extract FaceDetailer results |
| Unpack Tags | ⬡ Unpack Tags | `UME_PIPELINE` | Tags data | Extract tag/metadata |
| Unpack Pipe | ⬡ Unpack Pipe | `UME_PIPELINE` | Full decomposition | Extract all pipeline internals |

!!! tip "When to use Pack/Unpack"
    - **Pack** → when you want to use standard ComfyUI loader nodes but feed into the UmeAiRT KSampler
    - **Unpack** → when you want to feed UmeAiRT pipeline data into standard ComfyUI post-processing nodes

<!-- TODO: Screenshot — Pack Bundle node converting standard outputs into UME_BUNDLE -->
<!-- PLACEHOLDER: Show standard ComfyUI CheckpointLoaderSimple → Pack Bundle → UmeAiRT KSampler workflow -->
