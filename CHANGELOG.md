# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- Removed `hf_token` STRING input from `BundleLoader` to prevent token exposure in workflow JSON files. Token is now read automatically from `HF_TOKEN` env var or `~/.cache/huggingface/token`.
- Logs a helpful message with HuggingFace link when no token is found.

### Changed

- **Prompt Caching**: Enhanced `BlockSampler` "Fast Start" caching to be explicitly LoRA-aware, preserving ~30s performance gains while safely recompiling when upstream modifiers change.
- **Type Hinting**: Added strict Python type hints to core processor methods across `block_sampler.py`, `block_loaders.py`, and `block_inputs.py` to formally enforce `common.py` bundle contracts (`UME_BUNDLE`, `UME_SETTINGS`, `UME_IMAGE`, `UME_PIPELINE`).
- **Code Cleanup**: Removed unused legacy pipeline imports from `block_sampler.py` and `logic_nodes.py`.

### Fixed

- **Silent Exceptions**: Refactored multiple generic `except Exception: pass` blocks in `block_loaders.py` and `logic_nodes.py` to properly surface warnings (via `log_node(color="YELLOW")`) regarding missing HF tokens, missing Text Encoders, GGUF failures, and unavailable internals.

### Changed

- **Wireless → Pipeline Rename**: Renamed all 11 `Wireless*` classes and `NODE_CLASS_MAPPINGS` keys to `Pipeline*` (e.g., `UmeAiRT_WirelessImageSaver` → `UmeAiRT_PipelineImageSaver`).
- **Category Normalization**: Standardized all node categories to `UmeAiRT/Block/*`, `UmeAiRT/Pipeline/*`, `UmeAiRT/Utils/*` hierarchy.
- **DRY: Outpaint code**: Extracted ~40 lines of duplicated outpaint padding logic into `apply_outpaint_padding()` in `common.py`.
- **DRY: Prompt encoding**: Centralized inline CLIP prompt encoding into `encode_prompts()` in `common.py`.
- **Input/Output Consistency**: Renamed `pipeline` input parameter to `generation` across all post-processing nodes for consistency with the BlockSampler output name.
- **Display Name Cleanup**: Removed `(Block)`, `(Simple)`, and `(Pipeline)` suffixes from all node display names. Block Sampler renamed to KSampler.
- **Modular Split**: Split monolithic `block_nodes.py` (1426 lines) into 3 focused sub-modules: `block_inputs.py` (LoRA, ControlNet, Settings, Image, Prompts), `block_loaders.py` (Model Loaders, BundleAutoLoader), `block_sampler.py` (BlockSampler). `block_nodes.py` is now a re-export shim.
- **DRY: Bundle download helpers**: Extracted `_get_bundle_dropdowns()` and `_download_bundle_files()` shared helpers — used by both `BundleLoader` and `Bundle_Downloader`.

### Added

- `TypedDict` type definitions: `UmeBundle`, `UmeSettings`, `UmeImage` in `common.py`.
- `encode_prompts()` and `apply_outpaint_padding()` utility functions in `common.py`.
- `_get_hf_token()` helper for secure HuggingFace token retrieval.
- `tests/test_common.py`: 13 unit tests for core common.py components.
- **Bundle Downloader** (`UmeAiRT_Bundle_Downloader`): Standalone download utility — downloads model bundles to correct ComfyUI folders without loading into VRAM. Ideal for RunPod/cloud pre-downloading.

### Removed

- Deleted `UmeAiRT_BlockUltimateSDUpscale` and `UmeAiRT_BlockFaceDetailer` — duplicate of Pipeline equivalents with identical `UME_PIPELINE` interface.
- Removed legacy "Wireless" aliases from `NODE_CLASS_MAPPINGS`.

### Fixed

- Fixed `test_traversal.py` broken by removed `UME_SHARED_STATE`. Rewritten with 6 security test cases.
- Fixed 4 silent `except Exception: pass` — now log errors via `log_node()`.
- Cleaned `umeairt_colors.js`: removed ~30 phantom entries, fixed duplicate `UME_BUNDLE` slot color.
- Updated `AGENTS.md` and `README.md`: removed outdated references, documented new architecture.
- Fixed `NameError: pipeline` in `ImageSaver.save_images()` — missed reference during `pipeline` → `generation` rename.

### Changed (Architecture Refactoring)

- **Hub-and-Spoke Pipeline**: The `BlockSampler` is now the central hub that creates the `GenerationContext` (`UME_PIPELINE`). Loaders and settings nodes feed into it as side-inputs.
- **Loaders → `UME_BUNDLE`**: All 6 loader nodes (Checkpoint, FLUX, Fragmented, ZIMG, Advanced, BundleLoader) now return a single `UME_BUNDLE` dict `{model, clip, vae, model_name}` instead of three separate outputs.
- **`GenerationSettings` → `UME_SETTINGS`**: Returns a settings dict instead of requiring a pipeline input. No longer creates a `GenerationContext`.
- **`BlockSampler`**: Accepts `model_bundle` (UME_BUNDLE) + `settings` (UME_SETTINGS) as inputs, creates `GenerationContext` internally, stores sampled image within it, returns `UME_PIPELINE`.
- **Post-process nodes → pipeline-only**: All 8 post-processing nodes (UltimateUpscale Simple/Advanced, SeedVR2 Simple/Advanced, FaceDetailer Simple/Advanced, Detail Daemon Simple/Advanced) now read the image from `pipeline.image` and return `UME_PIPELINE` with the updated image. No more separate image input/output.
- **`ImageSaver` → pipeline-only**: Reads image from `pipeline.image` instead of a separate `images` input.
- **`BlockImageProcess`**: Removed `pipeline` input dependency. Added `auto_resize` flag that is stored in the `UME_IMAGE` bundle and acted upon by the `BlockSampler` using generation settings dimensions.
- **Display names**: Loader outputs renamed to `model_bundle`, sampler/post-process outputs renamed to `generation`.
- **`GenerationContext`**: Added `image` field, renamed `sampler` → `sampler_name`, `positive` → `positive_prompt`, `negative` → `negative_prompt`.

### Added

- Custom connection colors for `UME_BUNDLE` (#3498DB, bright blue) and `UME_PIPELINE` (#1ABC9C, teal) in `web/umeairt_colors.js`.
- **New node: `Unpack Pipeline`** — Decomposes a `UME_PIPELINE` into 15 native ComfyUI outputs (IMAGE, MODEL, CLIP, VAE, prompts, settings, denoise) for full interoperability with native and community nodes.
- Updated `Unpack Image Bundle` to output all 5 fields: image, mask, mode, denoise, auto_resize (previously only image and mask).
- **New node: `Pack Models Bundle`** — Packs native MODEL, CLIP, VAE into a `UME_BUNDLE` for use with Block nodes. Enables interoperability from any native or community loader into the UmeAiRT pipeline.

### Fixed

- Fixed critical installation issues by synchronizing `pyproject.toml` dependencies with `requirements.txt`.
- Removed duplicated and outdated class definitions (`UmeAiRT_FilesSettings_FLUX`, `UmeAiRT_FilesSettings_Fragmented`) in `modules/block_nodes.py`.
- Fixed manifest loading bug by correcting `bundles.json` reference to `umeairt_bundles.json` in `modules/utils_nodes.py`.
- Replaced numerous bare `except: pass` statements across the codebase with specific or generic exception handling to improve debuggability and stability.
- Restored missing activation switches (`lora_{i}_on`) in all `UmeAiRT_LoraBlock` nodes to properly toggle LoRAs on or off.
- Fixed `UmeAiRT_FilesSettings_Checkpoint_Advanced` incorrectly returning `UME_PIPELINE` instead of `UME_BUNDLE`, making it incompatible with the `BlockSampler`. Now returns a standard `UME_BUNDLE` dict like all other loaders.
- Fixed `UmeAiRT_Unpack_FilesBundle` accepting obsolete `UME_FILES` type; now accepts `UME_BUNDLE`.
- Fixed `UmeAiRT_Unpack_Settings` reading `sampler` key instead of `sampler_name`, causing it to always return the default `"euler"`.

### Removed

- Removed `UmeAiRT_WirelessKSampler` from `__init__.py` registrations (class was already deleted from `logic_nodes.py`, causing a latent `ImportError` at startup).
- Removed orphaned `UME_SHARED_STATE[KEY_LORAS]` write from `MultiLoraLoader` (Block nodes no longer read from global state).

### Added

- Added automated `tests/test_smoke.py` for validating core module imports and node class mappings.
- Implemented a startup "Health Check" node (or process) to validate dependencies and optimizations.
- Added `tests/test_traversal.py` for path traversal security regression testing.

### Security

- Added defense-in-depth path traversal guard in `ImageSaverLogic.save_images()` (`modules/image_saver_core/logic.py`). The output path is now validated with `os.path.abspath()` + `startswith()` to ensure it stays within the output directory, independently of caller-side sanitization.
