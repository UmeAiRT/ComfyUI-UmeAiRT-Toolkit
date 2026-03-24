"""Model manifest loading, caching, and bundle resolution.

Fetches model_manifest.json from the UmeAiRT Assets repo (HuggingFace),
caches locally, and provides bundle dropdown population and file resolution
for the Bundle Auto-Loader.
"""
import os
import json
import urllib.request
import folder_paths
from .common import log_node
from .download_utils import get_hf_token, download_file


# Maps path_type values from model_manifest.json to ComfyUI folder names
PATH_TYPE_TO_FOLDERS = {
    # Diffusion models
    "flux_diff":       ["diffusion_models"],
    "flux_unet":       ["unet"],
    "zimg_diff":       ["diffusion_models"],
    "zimg_unet":       ["unet"],
    "wan_diff":        ["diffusion_models"],
    "hidream_diff":    ["diffusion_models"],
    "qwen_diff":       ["diffusion_models"],
    "ltxv_diff":       ["diffusion_models"],
    "ltxv_ckpt":       ["checkpoints"],
    "ltx2_diff":       ["diffusion_models"],
    # Text encoders
    "clip":            ["clip", "text_encoders"],
    "text_encoders_t5":    ["clip", "text_encoders"],
    "text_encoders_qwen":  ["clip", "text_encoders"],
    "text_encoders_gemma": ["clip", "text_encoders"],
    "text_encoders_llama": ["clip", "text_encoders"],
    "text_encoders_ltx":   ["clip", "text_encoders"],
    # Vision / VAE / Other
    "clip_vision":     ["clip_vision"],
    "vae":             ["vae"],
    "latent_upscale":  ["upscale_models"],
    "melband":         ["custom"],
}


def find_file_in_folders(filename, folder_types):
    """Search for a file across multiple ComfyUI folder types by filename only.

    Most users dump files at the root of the category folder, so we search
    by filename regardless of subdirectory structure.

    If a .aria2 or .download control file exists alongside the file, the
    previous download was interrupted — the file is considered incomplete.

    Args:
        filename (str): The filename to search for.
        folder_types (list[str]): ComfyUI folder type names to search in.

    Returns:
        str or None: The full path if found and complete, otherwise None.
    """
    for folder_type in folder_types:
        try:
            path = folder_paths.get_full_path(folder_type, filename)
            if path and os.path.exists(path):
                # Check for interrupted download markers
                if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                    log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                    return None
                return path
        except Exception as e:
            log_node(f"Bundle Loader: Error searching in '{folder_type}': {e}", color="YELLOW")
        # Also try GGUF-specific folders
        if folder_type == "unet":
            try:
                path = folder_paths.get_full_path("unet_gguf", filename)
                if path and os.path.exists(path):
                    if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                        log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                        return None
                    return path
            except Exception as e:
                log_node(f"Bundle Loader: unet_gguf lookup failed for '{filename}': {e}", color="YELLOW")
        if folder_type == "clip":
            try:
                path = folder_paths.get_full_path("clip_gguf", filename)
                if path and os.path.exists(path):
                    if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                        log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                        return None
                    return path
            except Exception as e:
                log_node(f"Bundle Loader: clip_gguf lookup failed for '{filename}': {e}", color="YELLOW")
    return None


def get_download_dest(filename, folder_type):
    """Get the download destination path (root of the first registered folder).

    Args:
        filename (str): The target filename.
        folder_type (str): The primary ComfyUI folder type name.

    Returns:
        str: The absolute path where the file should be downloaded.
    """
    try:
        paths = folder_paths.get_folder_paths(folder_type)
        if paths:
            dest_dir = paths[0]
            os.makedirs(dest_dir, exist_ok=True)
            return os.path.join(dest_dir, filename)
    except Exception as e:
        log_node(f"Bundle Loader: Could not resolve folder path for '{folder_type}': {e}", color="YELLOW")
    # Fallback: models/<folder_type>/
    fallback = os.path.join(folder_paths.models_dir, folder_type)
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, filename)


# --- Manifest Loading & Caching ---

_MANIFEST_CACHE = None
_MANIFEST_URL = "https://huggingface.co/UmeAiRT/ComfyUI-Auto-Installer-Assets/resolve/main/models/model_manifest.json"


def _get_manifest_cache_path():
    """Return local cache path for the remote manifest."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "umeairt")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "model_manifest.json")


def load_manifest():
    """Load the model manifest, fetching from remote if stale.

    Priority:
    1. In-memory cache (fastest)
    2. Remote fetch from HuggingFace (if cache is missing or >24h old)
    3. Local cache file (if remote fetch fails)
    4. Fallback to bundled umeairt_bundles.json (legacy)

    Returns:
        dict: The parsed manifest data.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    import time
    cache_path = _get_manifest_cache_path()
    cache_max_age = 24 * 60 * 60  # 24 hours

    # Check if local cache is fresh enough
    need_fetch = True
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < cache_max_age:
            need_fetch = False

    # Try remote fetch
    if need_fetch:
        try:
            hf_token = get_hf_token()
            headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            req = urllib.request.Request(_MANIFEST_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
                data = json.loads(raw)
                # Write to cache
                with open(cache_path, 'wb') as f:
                    f.write(raw)
                log_node("Bundle Loader: 📡 Model manifest updated from remote.", color="GREEN")
                _MANIFEST_CACHE = data
                return data
        except Exception as e:
            log_node(f"Bundle Loader: Remote manifest fetch failed: {e}", color="YELLOW")

    # Read from local cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                _MANIFEST_CACHE = json.load(f)
                log_node("Bundle Loader: 📦 Using cached model manifest.", color="CYAN")
                return _MANIFEST_CACHE
        except Exception as e:
            log_node(f"Bundle Loader: Cache read failed: {e}", color="YELLOW")

    # Final fallback: legacy bundled file
    legacy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umeairt_bundles.json")
    if os.path.exists(legacy_path):
        log_node("Bundle Loader: ⚠️ Using legacy umeairt_bundles.json fallback.", color="YELLOW")
        with open(legacy_path, 'r', encoding='utf-8') as f:
            _MANIFEST_CACHE = json.load(f)
            return _MANIFEST_CACHE

    return {}


def get_bundle_dropdowns():
    """Return (categories, versions) lists from model manifest for dropdown population.

    Categories are FAMILY/VARIANT pairs (e.g. 'FLUX/Dev', 'Z-IMAGE/Turbo').
    For legacy bundles (flat structure), categories are top-level keys.
    """
    data = load_manifest()
    categories = []
    all_versions = set()

    for family_key, family_data in data.items():
        if family_key.startswith("_"):
            continue
        if not isinstance(family_data, dict):
            continue

        # Detect manifest v3 (has _family_meta) vs legacy (has _meta directly)
        if "_family_meta" in family_data:
            # Manifest v3: FAMILY → VARIANT → version
            for variant_key, variant_data in family_data.items():
                if variant_key.startswith("_") or not isinstance(variant_data, dict):
                    continue
                cat_label = f"{family_key}/{variant_key}"
                categories.append(cat_label)
                for ver_key in variant_data.keys():
                    if ver_key != "_meta":
                        all_versions.add(ver_key)
        else:
            # Legacy: CATEGORY → version (fallback for umeairt_bundles.json)
            categories.append(family_key)
            for ver_key in family_data.keys():
                if ver_key != "_meta":
                    all_versions.add(ver_key)

    if not categories:
        categories = ["No Bundles Found"]
    versions_list = sorted(list(all_versions)) if all_versions else ["Select Category First"]
    return categories, versions_list


def download_bundle_files(category, version):
    """Download all files for a bundle, skipping already-present ones.

    Supports both manifest v3 ('FAMILY/VARIANT' categories) and legacy flat categories.

    Returns:
        tuple: (resolved_files dict, meta dict, downloaded count, skipped count, errors list)
    """
    hf_token = get_hf_token()
    if not hf_token:
        log_node(
            "💡 No HF token found. To speed up downloads, create a token at "
            "https://huggingface.co/settings/tokens and set HF_TOKEN in your environment variables.",
            color="YELLOW"
        )

    data = load_manifest()

    # Resolve category to the right level in the manifest
    if "/" in category:
        # Manifest v3: FAMILY/VARIANT
        family_key, variant_key = category.split("/", 1)
        if family_key not in data:
            raise ValueError(f"Family '{family_key}' not found in manifest.")
        family_data = data[family_key]
        if variant_key not in family_data:
            raise ValueError(f"Variant '{variant_key}' not found for {family_key}.")
        variant_data = family_data[variant_key]
        meta = variant_data.get("_meta", {})
        # Base URL from top-level _sources (prefer huggingface)
        sources = data.get("_sources", {})
        base_url = sources.get("huggingface", "")
    else:
        # Legacy flat structure
        if category not in data:
            raise ValueError(f"Category '{category}' not found in manifest.")
        variant_data = data[category]
        meta = variant_data.get("_meta", {})
        base_url = meta.get("base_url", "")

    if version not in variant_data:
        raise ValueError(f"Version '{version}' not found for {category}.")
    bundle_def = variant_data[version]
    files = bundle_def.get("files", [])
    min_vram = bundle_def.get("min_vram", 0)
    log_node(f"📥 {category} / {version} ({len(files)} files, min VRAM: {min_vram}GB)")

    resolved_files = {}
    downloaded = 0
    skipped = 0
    errors = []

    for file_entry in files:
        pt = file_entry["path_type"]
        # Manifest v3 uses "path", legacy uses "filename" + "url"
        rel_path = file_entry.get("path", file_entry.get("url", ""))
        filename = os.path.basename(rel_path) if rel_path else file_entry.get("filename", "")
        expected_sha256 = file_entry.get("sha256", "")
        folder_types = PATH_TYPE_TO_FOLDERS.get(pt, [pt])
        local_path = find_file_in_folders(filename, folder_types)
        if local_path:
            log_node(f"  ✅ '{filename}' already present — skipping.", color="GREEN")
            skipped += 1
        else:
            try:
                full_url = f"{base_url}/{rel_path}" if not rel_path.startswith("http") else rel_path
                dest = get_download_dest(filename, folder_types[0])
                download_file(full_url, dest, hf_token=hf_token, expected_sha256=expected_sha256)
                downloaded += 1
            except Exception as e:
                log_node(f"  ❌ Failed to download '{filename}': {e}", color="RED")
                errors.append(filename)
        if pt not in resolved_files:
            resolved_files[pt] = []
        resolved_files[pt].append(filename)

    return resolved_files, meta, downloaded, skipped, errors
