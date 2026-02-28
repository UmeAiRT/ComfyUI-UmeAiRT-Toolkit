# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed critical installation issues by synchronizing `pyproject.toml` dependencies with `requirements.txt`.
- Removed duplicated and outdated class definitions (`UmeAiRT_FilesSettings_FLUX`, `UmeAiRT_FilesSettings_Fragmented`) in `modules/block_nodes.py`.
- Fixed manifest loading bug by correcting `bundles.json` reference to `umeairt_bundles.json` in `modules/utils_nodes.py`.
- Replaced numerous bare `except: pass` statements across the codebase with specific or generic exception handling to improve debuggability and stability.
- Restored missing activation switches (`lora_{i}_on`) in all `UmeAiRT_LoraBlock` nodes to properly toggle LoRAs on or off.

### Added

- Added automated `tests/test_smoke.py` for validating core module imports and node class mappings.
- Implemented a startup "Health Check" node (or process) to validate dependencies and optimizations.
- Added `tests/test_traversal.py` for path traversal security regression testing.

### Security

- Added defense-in-depth path traversal guard in `ImageSaverLogic.save_images()` (`modules/image_saver_core/logic.py`). The output path is now validated with `os.path.abspath()` + `startswith()` to ensure it stays within the output directory, independently of caller-side sanitization.
