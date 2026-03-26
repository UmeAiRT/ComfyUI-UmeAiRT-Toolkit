# TODO — UmeAiRT Toolkit

> Technical backlog. Items from the [critical analysis](docs/codemaps/structure.md).
> Convention: use `# TODO(UmeAiRT):` inline for code-level markers.

## Completed

- [x] **Refactor monolithic files (`block_loaders.py`)** — Download logic → `download_utils.py`, manifest → `manifest.py`
- [x] **SHA256 hash verification for downloads** — `verify_file_hash()` in `download_utils.py`
- [x] **Remote model manifest integration** — `load_manifest()` with fallback chain
- [x] **Dataclass bundles** — `UmeBundle`, `UmeSettings`, `UmeImage` migrated from `TypedDict`
- [x] **Coverage reporting in CI** — `coverage` job, threshold 40%
- [x] **CI test harness** — All tests run via `run_tests.py` with ComfyUI mocks
- [x] **MkDocs documentation site** — 20 node pages, architecture diagrams, auto-deploy to GitHub Pages
- [x] **Wildcard output types** — Unpack nodes use `*` type for sampler/scheduler (fixes COMBO connection issue)
- [x] **Split `logic_nodes.py`** — Split 811-line monolith into `upscale_nodes.py`, `seedvr2_nodes.py`, `face_nodes.py`, `detail_daemon_nodes.py` with re-export shim
- [x] **Extract `_load_diffusion_model()` helper** — DRY loader logic in `block_loaders.py` (FLUX/ZIMG/Bundle)

## Medium Priority

- [ ] **Add screenshots to documentation** — Each node page has `<!-- TODO -->` placeholders describing what to capture
- [ ] **Configure custom domain** — Set up `toolkit.umeai.art` CNAME for GitHub Pages
- [ ] **Add Z-IMG Loader to Bundle system** — Extend manifest categories for Lumina2 models

## Low Priority / Future

- [ ] **Increase coverage to 50%+** — Focus on `block_sampler.py` (need deep ComfyUI mocks)
- [ ] **Docs translations** — French translation of documentation
- [ ] **MkDocs CI strict check** — Add `mkdocs build --strict` validation to CI pipeline

