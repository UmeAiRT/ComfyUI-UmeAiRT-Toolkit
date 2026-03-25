# Architecture

## Block Pipeline

The UmeAiRT Toolkit replaces ComfyUI's traditional spaghetti wiring with a **block architecture**. Instead of connecting individual model/clip/vae/conditioning wires, you pass typed bundles:

| Bundle Type | Contents | Created By |
|-------------|----------|------------|
| `UME_BUNDLE` | model + clip + vae + model_name | Loader nodes |
| `UME_SETTINGS` | width, height, steps, cfg, sampler, scheduler, seed | Generation Settings |
| `UME_IMAGE` | image + mask + mode + denoise + controlnets | Image Loader/Process |
| `UME_LORA_STACK` | list of (name, model_strength, clip_strength) | LoRA Block nodes |
| `UME_PIPELINE` | Full generation context (all of the above + latent + result) | KSampler |

## Data Flow

```mermaid
graph TB
    subgraph "Model Loading"
        CKP["⬡ Checkpoint Loader"]
        FLUX["⬡ FLUX Loader"]
        ZIMG["⬡ Z-IMG Loader"]
        BDL["⬡ Bundle Auto-Loader"]
    end

    subgraph "Configuration"
        SET["⬡ Generation Settings"]
        POS["⬡ Positive Prompt"]
        NEG["⬡ Negative Prompt"]
        LORA["⬡ LoRA Block"]
        IMG["⬡ Image Loader"]
        IMGP["⬡ Image Process"]
        CNET["⬡ ControlNet Apply"]
    end

    subgraph "Sampling"
        KS["⬡ KSampler"]
    end

    subgraph "Post-Process"
        UP["⬡ UltimateSD Upscale"]
        SVR["⬡ SeedVR2 Upscale"]
        FD["⬡ FaceDetailer"]
    end

    subgraph "Output"
        SAV["⬡ Image Saver"]
        SRC["⬡ Source Image Output"]
        INP["⬡ Inpaint Composite"]
    end

    CKP -->|UME_BUNDLE| KS
    FLUX -->|UME_BUNDLE| KS
    ZIMG -->|UME_BUNDLE| KS
    BDL -->|UME_BUNDLE| KS
    SET -->|UME_SETTINGS| KS
    POS -->|POSITIVE| KS
    NEG -->|NEGATIVE| KS
    LORA -->|UME_LORA_STACK| KS
    IMG -->|UME_IMAGE| IMGP
    IMGP -->|UME_IMAGE| CNET
    CNET -->|UME_IMAGE| KS

    KS -->|UME_PIPELINE| UP
    KS -->|UME_PIPELINE| SVR
    KS -->|UME_PIPELINE| FD
    UP -->|UME_PIPELINE| FD
    SVR -->|UME_PIPELINE| FD

    KS -->|UME_PIPELINE| SAV
    UP -->|UME_PIPELINE| SAV
    FD -->|UME_PIPELINE| SAV
    SVR -->|UME_PIPELINE| SAV
```

## Module Structure

```
ComfyUI-UmeAiRT-Toolkit/
├── __init__.py              # Node registration (47 nodes)
├── modules/
│   ├── block_loaders.py     # Model loading nodes
│   ├── block_inputs.py      # LoRA, ControlNet, Settings, Image, Prompts
│   ├── block_sampler.py     # Central KSampler hub
│   ├── logic_nodes.py       # Upscale, FaceDetailer, Detailer Daemon
│   ├── image_nodes.py       # Source Image, Inpaint Composite, Image Saver
│   ├── model_nodes.py       # Multi-LoRA Loader (wired)
│   ├── utils_nodes.py       # Label, Downloader, Unpack, Health Check
│   ├── common.py            # Shared dataclasses and utilities
│   ├── manifest.py          # Model manifest parsing
│   ├── download_utils.py    # Download engine (aria2c + urllib)
│   ├── extra_samplers.py    # Custom sampler registration
│   └── optimization_utils.py # VRAM management, SageAttention
├── web/                     # Frontend JS (widget extensions)
├── docs/                    # This documentation
└── tests/                   # Test suite (140+ tests)
```

## Bundle Auto-Download

The Bundle system uses a remote `model_manifest.json` hosted on [UmeAiRT Assets](https://github.com/UmeAiRT/ComfyUI-Auto-Installer-Assets):

```mermaid
sequenceDiagram
    participant User
    participant BundleLoader as ⬡ Bundle Auto-Loader
    participant Manifest as model_manifest.json
    participant HF as HuggingFace

    User->>BundleLoader: Select category + version
    BundleLoader->>Manifest: Fetch manifest (cached)
    Manifest-->>BundleLoader: File list + SHA256 hashes
    BundleLoader->>BundleLoader: Check local files
    alt Files missing
        BundleLoader->>HF: Download via aria2c/urllib
        HF-->>BundleLoader: Model files
        BundleLoader->>BundleLoader: Verify SHA256
    end
    BundleLoader->>BundleLoader: Load model/clip/vae
    BundleLoader-->>User: UME_BUNDLE ready
```
