# TODO — UmeAiRT Toolkit

> Technical backlog. Items from the [critical analysis](docs/codemaps/structure.md).
> Convention: use `# TODO(UmeAiRT):` inline for code-level markers.

## High Priority

- [ ] **Refactor monolithic files (`block_loaders.py`)**
  - Separate ComfyUI node definitions from backend/network logic (e.g., move download systems to `network_utils.py` or a dedicated service module).
- [ ] **Rethink aggressive VRAM management**
  - Replace manual cache clearing (`gc.collect()`, `mm.free_memory()`) in `logic_nodes.py` (e.g. SeedVR2 nodes) with ComfyUI's native VRAM allocation hooks to prevent interfering with the ecosystem's cache manager.
- [ ] **SHA256 hash verification for downloads**
  - Add `_verify_file_hash(path, expected_sha256)` helper to `block_loaders.py`
  - Add `sha256` fields to `umeairt_bundles.json` manifest
  - Verify after both aria2c and urllib downloads
  - 📍 Inline marker: `block_loaders.py` L599

## Medium Priority

- [x] **Coverage reporting in CI** — Added `coverage` job to `ci.yml` (Python 3.12, fail-under=30%)
- [ ] **Document download timeouts** — Add `timeout` parameter to `_download_with_urllib()` with tooltip

## Low Priority / Future

- [ ] **Consider `dataclass` for bundles** — Replace `TypedDict` with `@dataclass` for `UmeBundle`/`UmeSettings` (adds default values, `__post_init__` validation)
- [ ] **Centralize `process_and_stitch` import** — Currently imported inline in SeedVR2 nodes; move to top when stable
- [ ] **Integration tests** — Test full pipeline: Loader → Sampler → PostProcess → Saver with mocked ComfyUI nodes
