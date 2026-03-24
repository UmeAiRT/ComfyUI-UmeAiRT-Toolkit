# TODO — UmeAiRT Toolkit

> Technical backlog. Items from the [critical analysis](docs/codemaps/structure.md).
> Convention: use `# TODO(UmeAiRT):` inline for code-level markers.

## High Priority

- [ ] **Refactor monolithic files (`block_loaders.py`)**
  - Separate ComfyUI node definitions from backend/network logic (e.g., move download systems to `network_utils.py` or a dedicated service module).
- [ ] **Rethink aggressive VRAM management**
  - Replace manual cache clearing (`gc.collect()`, `mm.free_memory()`) in `logic_nodes.py` (e.g. SeedVR2 nodes) with ComfyUI's native VRAM allocation hooks to prevent interfering with the ecosystem's cache manager.
- [x] **SHA256 hash verification for downloads**
  - Added `_verify_file_hash(path, expected_sha256)` helper to `block_loaders.py`
  - Hashes sourced from `model_manifest.json` (fetched from Assets repo, cached 24h)
  - Verified after both aria2c and urllib downloads
- [x] **Remote model manifest integration**
  - `_load_manifest()` fetches `model_manifest.json` from HuggingFace at startup
  - Dropdown categories now use `FAMILY/VARIANT` format (e.g. `FLUX/Dev`, `WAN_2.1/T2V`)
  - Fallback chain: remote → local cache → legacy `umeairt_bundles.json`

## Medium Priority

- [x] **Coverage reporting in CI** — Added `coverage` job to `ci.yml` (Python 3.12, fail-under=30%)
- [x] **Document download timeouts** — Added configurable `timeout` parameter to `_download_with_urllib()` (default: 300s, was hardcoded 60s)

## Low Priority / Future

- [ ] **Consider `dataclass` for bundles** — Replace `TypedDict` with `@dataclass` for `UmeBundle`/`UmeSettings` (adds default values, `__post_init__` validation)
- [x] **Centralize `process_and_stitch` import** — Moved to `_get_seedvr2_modules()` lazy loader in `logic_nodes.py`
- [ ] **Integration tests** — Test full pipeline: Loader → Sampler → PostProcess → Saver with mocked ComfyUI nodes
