# TODO — UmeAiRT Toolkit

> Technical backlog. Items from the [critical analysis](docs/codemaps/structure.md).
> Convention: use `# TODO(UmeAiRT):` inline for code-level markers.

## High Priority

- [x] **Refactor monolithic files (`block_loaders.py`)**
  - Download logic extracted to `download_utils.py` (aria2c, urllib, SHA256 verify)
  - Manifest logic extracted to `manifest.py` (load, cache, bundle resolution)
  - `block_loaders.py` reduced to ~350 lines of pure ComfyUI node definitions
  - Re-exports and legacy aliases kept for backward compatibility
- [x] **Rethink aggressive VRAM management**
  - Removed manual `gc.collect()` / `mm.free_memory()` calls from `logic_nodes.py`
- [x] **SHA256 hash verification for downloads**
  - `verify_file_hash(path, expected_sha256)` in `download_utils.py`
  - Hashes sourced from `model_manifest.json` (fetched from Assets repo, cached 24h)
  - Verified after both aria2c and urllib downloads
- [x] **Remote model manifest integration**
  - `load_manifest()` in `manifest.py` fetches from HuggingFace at startup
  - Dropdown categories now use `FAMILY/VARIANT` format (e.g. `FLUX/Dev`, `WAN_2.1/T2V`)
  - Fallback chain: remote → local cache → legacy `umeairt_bundles.json`

## Medium Priority

- [x] **Coverage reporting in CI** — Added `coverage` job to `ci.yml` (Python 3.12, fail-under=30%)
- [x] **Document download timeouts** — Added configurable `timeout` parameter to `_download_with_urllib()` (default: 300s, was hardcoded 60s)

## Low Priority / Future

- [x] **Consider `dataclass` for bundles** — Migrated `UmeBundle`, `UmeSettings`, `UmeImage` from `TypedDict` to `@dataclass` with defaults and validation; all consumers updated to attribute access
- [x] **Centralize `process_and_stitch` import** — Moved to `_get_seedvr2_modules()` lazy loader in `logic_nodes.py`
- [x] **Integration tests** — Added `test_integration.py` covering dataclass behavior, Pack/Unpack roundtrips, Settings flow, and full Loader→Sampler→Pipeline→Unpack pipeline
