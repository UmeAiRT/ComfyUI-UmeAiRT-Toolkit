import torch
import os
import json
import urllib.request
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, log_node
from .logger import logger, log_progress
from typing import Tuple, Dict, Any

def _get_hf_token() -> str:
    """Retrieve HuggingFace token from environment or cache file.

    Checks in order:
    1. HF_TOKEN environment variable
    2. ~/.cache/huggingface/token file

    Returns:
        str: The token string, or empty string if not found.
    """
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token

    # 2. HuggingFace cache file
    hf_token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
    if os.path.isfile(hf_token_path):
        try:
            with open(hf_token_path, 'r', encoding='utf-8') as f:
                token = f.read().strip()
            if token:
                return token
        except Exception as e:
            log_node(f"Bundle Loader: Could not read HF token from cache: {e}", color="YELLOW")

    return ""



# --- Files / Model Loaders (Block) ---

class UmeAiRT_FilesSettings_Checkpoint:
    """Basic Loader for Standard Checkpoints. Returns raw MODEL, CLIP, VAE."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Choose the AI model file (.safetensors, .ckpt) for image generation."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load"
    CATEGORY = "UmeAiRT/Block/Loaders"
    
    def load(self, ckpt_name: str) -> Tuple[Dict[str, Any]]:
        model, clip, vae = comfy_nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        log_node(f"Checkpoint Loaded: {ckpt_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": ckpt_name},)

class UmeAiRT_FilesSettings_Checkpoint_Advanced:
    """
    Advanced version - loads Model, CLIP, and VAE from Checkpoint.
    Allows VAE override and CLIP Skip control.
    Best for SD1.5 and SDXL.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Select the Checkpoint model to load."}),
            },
            "optional": {
                 "vae_name": (["Baked"] + folder_paths.get_filename_list("vae"), {"tooltip": "Optional: Override the VAE."}),
                 "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1, "tooltip": "Optional: Set CLIP Skip layer."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Block/Loaders"

    def load_files(self, ckpt_name: str, vae_name: str = "Baked", clip_skip: int = -1) -> Tuple[Dict[str, Any]]:
        """Loads a checkpoint with optional VAE override and CLIP skip.

        Args:
            ckpt_name (str): The checkpoint filename.
            vae_name (str, optional): The VAE filename. Defaults to "Baked".
            clip_skip (int, optional): The CLIP skip layer. Defaults to -1.

        Returns:
            tuple: A tuple containing the bundled models.
        """
        # 1. Load Checkpoint
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        clip = out[1]
        vae = out[2] # Baked VAE

        # 2. Optional VAE Override
        if vae_name != "Baked":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 3. CLIP Skip Logic
        if clip_skip != -1:
             clip = clip.clone()
             clip.clip_layer(clip_skip)

        # 4. Return Bundle
        log_node(f"Checkpoint Advanced Loaded: {ckpt_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": ckpt_name},)


class UmeAiRT_FilesSettings_FLUX:
    """
    Bundles Model (UNET), CLIP, and VAE (Loaded Separately).
    Updates Global Wireless State.
    Best for FLUX and complex workflows.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"), {"tooltip": "Select the UNET model file."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"], {"tooltip": "Weight data type for loading the UNET (affects VRAM usage)."}),
                "clip_name1": (folder_paths.get_filename_list("clip"), {"tooltip": "Select the primary CLIP model (e.g., t5xxl)."}),
                "clip_name2": (folder_paths.get_filename_list("clip"), {"tooltip": "Select the secondary CLIP model (e.g., clip_l)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "Select the VAE model file."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Block/Loaders"

    def load_files(self, unet_name: str, weight_dtype: str, clip_name1: str, clip_name2: str, vae_name: str) -> Tuple[Dict[str, Any]]:
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"))

        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        log_node(f"FLUX Loaded: {unet_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": unet_name},)


class UmeAiRT_FilesSettings_Fragmented:
    """
    Fragmented Model Loader (Z-IMG style).
    Loads Model, CLIP, and VAE from separate files/folders.
    Model list combines 'checkpoints', 'diffusion_models', and 'unet'.
    CLIP list combines 'clip' and 'text_encoders' folders.
    """
    @classmethod
    def INPUT_TYPES(s):
        # 1. Get Models (Checkpoints + Diffusion Models + UNET)
        ckpts = folder_paths.get_filename_list("checkpoints")
        diff_models = folder_paths.get_filename_list("diffusion_models")
        unets = folder_paths.get_filename_list("unet")
        
        # Combine and deduplicate
        all_models = sorted(list(set(ckpts + diff_models + unets)))

        # 2. Get CLIPs (Standard + Text Encoders)
        clips = folder_paths.get_filename_list("clip")
        try:
            tes = folder_paths.get_filename_list("text_encoders")
            if tes:
                clips = sorted(list(set(clips + tes)))
        except Exception as e:
            from .logger import log_node
            log_node(f"Fragmented Loader: Failed to get text_encoders list: {e}", color="YELLOW")
            pass
            
        # 3. Get VAEs
        vaes = folder_paths.get_filename_list("vae")
        
        return {
            "required": {
                "model_name": (all_models, {"tooltip": "Select Model (Checkpoint, Diffusion Model, or UNET)."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"tooltip": "Weight data type for loading the model."}),
                "clip_name": (clips, {"tooltip": "Select CLIP model (Text Encoder)."}),
                "clip_type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis"], {"tooltip": "Specify the CLIP model architecture."}),
                "vae_name": (vaes, {"tooltip": "Select VAE model."}),
            },
            "optional": {
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1, "tooltip": "CLIP Skip layer."}),
                "device": (["default", "cpu"], {"advanced": True, "tooltip": "Device to load the model on (default is GPU)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Block/Loaders"

    def load_files(self, model_name: str, clip_name: str, vae_name: str, weight_dtype: str = "default", clip_type: str = "stable_diffusion", clip_skip: int = -1, device: str = "default") -> Tuple[Dict[str, Any]]:
        """Loads components from multiple distinct folders with explicit typing."""
        ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
        diff_path = folder_paths.get_full_path("diffusion_models", model_name)
        unet_path = folder_paths.get_full_path("unet", model_name)
        model = None
        if diff_path or unet_path:
            final_path = diff_path if diff_path else unet_path
            model_options = {}
            if weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast": model_options["dtype"] = torch.float8_e4m3fn; model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2": model_options["dtype"] = torch.float8_e5m2
            model = comfy.sd.load_diffusion_model(final_path, model_options=model_options)
        elif ckpt_path:
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model = out[0]
        else:
            raise ValueError(f"Fragmented Loader: Model '{model_name}' not found.")
        
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if clip_path is None:
            try: clip_path = folder_paths.get_full_path("text_encoders", clip_name)
            except Exception as e:
                log_node(f"Fragmented Loader: text_encoders lookup failed: {e}", color="YELLOW")
        if clip_path is None:
            raise ValueError(f"Fragmented Loader: Could not find CLIP file '{clip_name}'.")
        clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_options = {}
        if device == "cpu": clip_options["load_device"] = clip_options["offload_device"] = torch.device("cpu")
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type_enum, model_options=clip_options)

        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        if clip_skip != -1:
             clip = clip.clone()
             clip.clip_layer(clip_skip)

        log_node(f"Fragmented Loaded: {model_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": model_name},)

class UmeAiRT_FilesSettings_ZIMG:
    """
    Z-IMG Specialized Loader (Simplified).
    - Only loads models from 'diffusion_models'.
    - Auto-detects weight_dtype (e4m3fn/e5m2) from filename.
    - Hardcoded CLIP Type: LUMINA2.
    """
    @classmethod
    def INPUT_TYPES(s):
        try:
            from ..vendor.comfyui_gguf import gguf_nodes
        except Exception as e:
            from .logger import log_node
            log_node(f"ZIMG Loader: Failed to import gguf_nodes: {e}", color="YELLOW")
            pass

        # 1. Get Models (Diffusion Models ONLY natively, plus GGUF)
        diff_models = folder_paths.get_filename_list("diffusion_models")
        if diff_models is None: diff_models = []
        try:
            unet_gguf = folder_paths.get_filename_list("unet_gguf")
            if unet_gguf: diff_models = diff_models + unet_gguf
        except Exception:
            pass
        diff_models = sorted(list(set(diff_models)))
        
        # 2. Get CLIPs (Standard + Text Encoders + GGUF)
        clips = folder_paths.get_filename_list("clip")
        if clips is None: clips = []
        try:
            tes = folder_paths.get_filename_list("text_encoders")
            if tes: clips = clips + tes
        except Exception:
            pass
        try:
            gguf_clips = folder_paths.get_filename_list("clip_gguf")
            if gguf_clips: clips = clips + gguf_clips
        except Exception:
            pass
        clips = sorted(list(set(clips)))
            
        # 3. Get VAEs
        vaes = folder_paths.get_filename_list("vae")
        
        return {
            "required": {
                "model_name": (diff_models, {"tooltip": "Select Diffusion Model (Z-IMG format)."}),
                "clip_name": (clips, {"tooltip": "Select CLIP model (Text Encoder)."}),
                "vae_name": (vaes, {"tooltip": "Select VAE model."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Block/Loaders"

    def load_files(self, model_name: str, clip_name: str, vae_name: str) -> Tuple[Dict[str, Any]]:
        """Loads models tailored for the Z-IMG format."""
        if model_name.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
            model = UnetLoaderGGUF().load_unet(model_name)[0]
        else:
            diff_path = folder_paths.get_full_path("diffusion_models", model_name)
            if not diff_path:
                raise ValueError(f"Z-IMG Loader: Model '{model_name}' not found.")
            model_options = {}
            if "e4m3fn" in model_name.lower(): model_options["dtype"] = torch.float8_e4m3fn
            elif "e5m2" in model_name.lower(): model_options["dtype"] = torch.float8_e5m2
            model = comfy.sd.load_diffusion_model(diff_path, model_options=model_options)

        if clip_name.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip = CLIPLoaderGGUF().load_clip(clip_name, type="lumina2")[0]
        else:
            clip_path = folder_paths.get_full_path("clip", clip_name)
            if clip_path is None:
                try: clip_path = folder_paths.get_full_path("text_encoders", clip_name)
                except Exception as e:
                    log_node(f"Z-IMG Loader: text_encoders lookup failed: {e}", color="YELLOW")
            if clip_path is None:
                raise ValueError(f"Z-IMG Loader: CLIP '{clip_name}' not found.")
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=comfy.sd.CLIPType.LUMINA2)

        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        log_node(f"Z-IMG Loaded: {model_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": model_name},)



# --- Bundle Auto-Loader ---

# Maps path_type values from umeairt_bundles.json to ComfyUI folder names
_PATH_TYPE_TO_FOLDERS = {
    "flux_diff": ["diffusion_models"],
    "flux_unet": ["unet"],
    "zimg_diff": ["diffusion_models"],
    "zimg_unet": ["unet"],
    "clip":      ["clip", "text_encoders"],
    "vae":       ["vae"],
}


def _find_file_in_folders(filename, folder_types):
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
            except Exception:
                pass
        if folder_type == "clip":
            try:
                path = folder_paths.get_full_path("clip_gguf", filename)
                if path and os.path.exists(path):
                    if os.path.exists(path + ".aria2") or os.path.exists(path + ".download"):
                        log_node(f"  ⚠️ '{filename}' has incomplete download — will resume.", color="YELLOW")
                        return None
                    return path
            except Exception:
                pass
    return None


def _get_download_dest(filename, folder_type):
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
    except Exception:
        pass
    # Fallback: models/<folder_type>/
    fallback = os.path.join(folder_paths.models_dir, folder_type)
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, filename)


def _find_aria2c():
    """Find aria2c executable, searching PATH and common Windows install locations.

    ComfyUI's embedded Python often doesn't inherit the user's system PATH,
    so we also search common installation directories.

    Returns:
        str or None: Full path to aria2c executable, or None if not found.
    """
    import shutil
    import subprocess

    # 0. Check for vendored binary bundled with the toolkit
    vendor_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor", "aria2")
    vendor_exe = os.path.join(vendor_dir, "aria2c.exe") if os.name == "nt" else os.path.join(vendor_dir, "aria2c")
    if os.path.isfile(vendor_exe):
        try:
            result = subprocess.run([vendor_exe, "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return vendor_exe
        except Exception:
            pass

    # 1. Try shutil.which (works if aria2c is on the current PATH)
    path = shutil.which("aria2c")
    if path:
        return path

    # 2. Search common Windows install locations
    if os.name == "nt":
        candidates = []
        home = os.path.expanduser("~")
        localappdata = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        programfiles = os.environ.get("ProgramFiles", r"C:\Program Files")
        programfilesx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

        # Scoop
        candidates.append(os.path.join(home, "scoop", "shims", "aria2c.exe"))
        candidates.append(os.path.join(home, "scoop", "apps", "aria2", "current", "aria2c.exe"))
        # Chocolatey
        candidates.append(r"C:\ProgramData\chocolatey\bin\aria2c.exe")
        # Standalone / manual install
        candidates.append(os.path.join(localappdata, "aria2", "aria2c.exe"))
        candidates.append(os.path.join(programfiles, "aria2", "aria2c.exe"))
        candidates.append(os.path.join(programfilesx86, "aria2", "aria2c.exe"))
        candidates.append(r"C:\aria2\aria2c.exe")
        # winget (typically goes into LOCALAPPDATA\Microsoft\WinGet\...)
        winget_dir = os.path.join(localappdata, "Microsoft", "WinGet", "Packages")
        if os.path.isdir(winget_dir):
            for d in os.listdir(winget_dir):
                if "aria2" in d.lower():
                    candidate = os.path.join(winget_dir, d, "aria2c.exe")
                    candidates.append(candidate)

        for candidate in candidates:
            if os.path.isfile(candidate):
                # Verify it actually runs
                try:
                    result = subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        return candidate
                except Exception:
                    pass

    # 3. Last resort: just try running it (maybe it's on a PATH we missed)
    try:
        result = subprocess.run(["aria2c", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "aria2c"
    except Exception:
        pass

    return None


# Cache: None = not checked yet, False = not available, str = path to aria2c
_ARIA2_PATH = None


def _download_with_aria2(url, dest_path, connections=8, hf_token=""):
    """Download a file using aria2c for multi-connection acceleration.

    Uses Popen to run aria2c in the background while polling file size
    to update ComfyUI's progress bar and log periodic progress.

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        connections (int): Number of parallel connections (default: 8).
        hf_token (str): Optional HuggingFace token for authentication.

    Returns:
        bool: True if download succeeded, False otherwise.
    """
    import subprocess
    import time
    filename = os.path.basename(dest_path)
    dest_dir = os.path.dirname(dest_path)

    # Get total file size via HEAD request for progress tracking
    total_size = 0
    try:
        headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception:
        pass

    size_mb = f" ({total_size / 1024 / 1024:.0f} MB)" if total_size else ""
    log_node(f"Bundle Loader: Downloading '{filename}'{size_mb} via aria2c ({connections} connections)...")

    try:
        aria2_exe = _ARIA2_PATH
        cmd = [
            aria2_exe,
            "--dir=" + dest_dir,
            "--out=" + filename,
            "--split=" + str(connections),
            "--max-connection-per-server=" + str(connections),
            "--min-split-size=1M",
            "--continue=true",
            "--file-allocation=none",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--console-log-level=notice",
            "--summary-interval=1",
            "--human-readable=true",
        ]
        if hf_token:
            cmd.extend(["--header", "Authorization: Bearer " + hf_token])
        cmd.append(url)

        import re
        # Regex to parse aria2c progress output: [#id 50MiB/100MiB(50%)]
        pct_re = re.compile(r'\((\d+)%\)')

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pbar = comfy.utils.ProgressBar(100)

        # Read aria2c output line by line in real-time
        for raw_line in iter(process.stdout.readline, b''):
            line = raw_line.decode(errors="ignore").strip()
            if not line:
                continue
            # Parse percentage from lines containing (XX%)
            match = pct_re.search(line)
            if match:
                pct = int(match.group(1))
                pbar.update_absolute(pct)
                log_progress(filename, pct)

        process.wait()
        returncode = process.returncode

        # Finalize progress bar line and UI bar
        log_progress(filename, 100, done=True)
        pbar.update_absolute(100)

        if returncode == 0 and os.path.exists(dest_path):
            # TODO(UmeAiRT): Add SHA256 hash verification here once umeairt_bundles.json has sha256 fields
            log_node(f"Bundle Loader: '{filename}' downloaded via aria2c.", color="GREEN")
            return True
        else:
            stderr = process.stderr.read().decode(errors="ignore").strip() if process.stderr else "no details"
            log_node(f"Bundle Loader: aria2c failed (code {returncode}: {stderr}), falling back to urllib.", color="YELLOW")
            return False

    except Exception as e:
        log_node(f"Bundle Loader: aria2c error: {e}, falling back to urllib.", color="YELLOW")
        return False



def _download_with_urllib(url, dest_path, hf_token=""):
    """Download a file with urllib and a ComfyUI progress bar (fallback).

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        hf_token (str): Optional HuggingFace token for authentication.
    """
    filename = os.path.basename(dest_path)
    temp_path = dest_path + ".download"
    log_node(f"Bundle Loader: Downloading '{filename}' via urllib...")

    headers = {"User-Agent": "ComfyUI-UmeAiRT-Toolkit"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        pbar = comfy.utils.ProgressBar(total_size) if total_size > 0 else None

        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(8192 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if pbar:
                    pbar.update_absolute(downloaded)

    # Rename temp file to final destination
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    log_node(f"Bundle Loader: '{filename}' downloaded successfully.", color="GREEN")


def _download_file(url, dest_path, hf_token=""):
    """Download a file, preferring aria2c for speed with urllib as fallback.

    Args:
        url (str): The full URL to download.
        dest_path (str): The local path to save to.
        hf_token (str): Optional HuggingFace token for authentication.

    Raises:
        RuntimeError: If all download methods fail.
    """
    global _ARIA2_PATH
    filename = os.path.basename(dest_path)

    try:
        # Try aria2c first (much faster for large model files)
        if _ARIA2_PATH is None:
            _ARIA2_PATH = _find_aria2c() or False
            if _ARIA2_PATH:
                log_node("Bundle Loader: aria2c detected — using accelerated downloads.", color="GREEN")
            else:
                hint = "Run: apt install aria2 (Linux) or bundled in vendor/aria2/ (Windows)" if os.name != "nt" else "bundled binary not found in vendor/aria2/"
                log_node(f"Bundle Loader: aria2c not found — using urllib. {hint}", color="YELLOW")

        if _ARIA2_PATH:
            if _download_with_aria2(url, dest_path, hf_token=hf_token):
                return

        # Fallback to urllib
        _download_with_urllib(url, dest_path, hf_token=hf_token)

    except Exception as e:
        # Clean up partial downloads
        for cleanup in [dest_path + ".download", dest_path + ".aria2"]:
            if os.path.exists(cleanup):
                try:
                    os.remove(cleanup)
                except Exception:
                    pass
        raise RuntimeError(f"Bundle Loader: Failed to download '{filename}': {e}")


_BUNDLES_JSON_CACHE = None

def _load_bundles_json():
    """Load and cache the umeairt_bundles.json manifest."""
    global _BUNDLES_JSON_CACHE
    if _BUNDLES_JSON_CACHE is not None:
        return _BUNDLES_JSON_CACHE
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umeairt_bundles.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            _BUNDLES_JSON_CACHE = json.load(f)
            return _BUNDLES_JSON_CACHE
    return {}


def _get_bundle_dropdowns():
    """Return (categories, versions) lists from umeairt_bundles.json for dropdown population."""
    data = _load_bundles_json()
    categories = [k for k in data.keys() if not k.startswith("_")] if data else ["No Bundles Found"]
    all_versions = set()
    for cat_key in categories:
        cat_data = data.get(cat_key, {})
        for ver_key in cat_data.keys():
            if ver_key != "_meta":
                all_versions.add(ver_key)
    versions_list = sorted(list(all_versions)) if all_versions else ["Select Category First"]
    return categories, versions_list


def _download_bundle_files(category, version):
    """Download all files for a bundle, skipping already-present ones.

    Returns:
        tuple: (resolved_files dict, meta dict, downloaded count, skipped count, errors list)
    """
    hf_token = _get_hf_token()
    if not hf_token:
        log_node(
            "💡 No HF token found. To speed up downloads, create a token at "
            "https://huggingface.co/settings/tokens and set HF_TOKEN in your environment variables.",
            color="YELLOW"
        )

    data = _load_bundles_json()
    if category not in data:
        raise ValueError(f"Category '{category}' not found in bundles manifest.")
    cat_data = data[category]
    meta = cat_data.get("_meta", {})
    base_url = meta.get("base_url", "")

    if version not in cat_data:
        raise ValueError(f"Version '{version}' not found for {category}.")
    bundle_def = cat_data[version]
    files = bundle_def.get("files", [])
    min_vram = bundle_def.get("min_vram", 0)
    log_node(f"📥 {category} / {version} ({len(files)} files, min VRAM: {min_vram}GB)")

    resolved_files = {}
    downloaded = 0
    skipped = 0
    errors = []

    for file_entry in files:
        pt = file_entry["path_type"]
        filename = file_entry["filename"]
        url_path = file_entry["url"]
        folder_types = _PATH_TYPE_TO_FOLDERS.get(pt, [pt])
        local_path = _find_file_in_folders(filename, folder_types)
        if local_path:
            log_node(f"  ✅ '{filename}' already present — skipping.", color="GREEN")
            skipped += 1
        else:
            try:
                full_url = base_url + url_path
                dest = _get_download_dest(filename, folder_types[0])
                _download_file(full_url, dest, hf_token=hf_token)
                downloaded += 1
            except Exception as e:
                log_node(f"  ❌ Failed to download '{filename}': {e}", color="RED")
                errors.append(filename)
        if pt not in resolved_files:
            resolved_files[pt] = []
        resolved_files[pt].append(filename)

    return resolved_files, meta, downloaded, skipped, errors


class UmeAiRT_BundleLoader:
    """Bundle Auto-Loader: select a model + version, auto-download missing files, and load them.

    Combines downloading and loading into one node.
    Reads from umeairt_bundles.json to populate dropdowns and determine loading strategy.
    """

    @classmethod
    def INPUT_TYPES(s):
        categories, versions_list = _get_bundle_dropdowns()
        return {
            "required": {
                "category": (categories, {"tooltip": "Select model family (e.g. FLUX, Z-IMAGE_TURBO)."}),
                "version": (versions_list, {"tooltip": "Select quantization/precision version (e.g. fp16, GGUF_Q4)."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_bundle"
    CATEGORY = "UmeAiRT/Block/Loaders"
    OUTPUT_NODE = True

    def load_bundle(self, category, version):
        """Download missing files and load the selected model bundle."""
        resolved_files, meta, downloaded, skipped, errors = _download_bundle_files(category, version)
        if errors:
            raise RuntimeError(f"Bundle Loader: {len(errors)} file(s) failed to download: {', '.join(errors)}")
        loader_type = meta.get("loader_type", "zimg")
        clip_type_str = meta.get("clip_type", "lumina2")

        model, clip, vae = None, None, None
        model_name = ""
        model_pt = None
        for pt_key in ["zimg_diff", "flux_diff", "zimg_unet", "flux_unet"]:
            if pt_key in resolved_files: model_pt = pt_key; break
        if model_pt:
            model_filename = resolved_files[model_pt][0]
            model_name = model_filename
            if model_filename.endswith(".gguf"):
                from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
                model = UnetLoaderGGUF().load_unet(model_filename)[0]
            else:
                model_path = _find_file_in_folders(model_filename, _PATH_TYPE_TO_FOLDERS.get(model_pt, ["diffusion_models"]))
                if not model_path: raise ValueError(f"Bundle Loader: Model '{model_filename}' not found.")
                model_options = {}
                ln = model_filename.lower()
                if "e4m3fn" in ln: model_options["dtype"] = torch.float8_e4m3fn
                elif "e5m2" in ln: model_options["dtype"] = torch.float8_e5m2
                model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)

        clip_files = resolved_files.get("clip", [])
        if clip_files:
            if loader_type == "flux" and len(clip_files) >= 2:
                clip_paths = [_find_file_in_folders(cf, ["clip", "text_encoders"]) for cf in clip_files]
                clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            else:
                cf = clip_files[0]
                if cf.endswith(".gguf"):
                    from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
                    clip = CLIPLoaderGGUF().load_clip(cf, type=clip_type_str)[0]
                else:
                    cp = _find_file_in_folders(cf, ["clip", "text_encoders"])
                    if not cp: raise ValueError(f"Bundle Loader: CLIP '{cf}' not found.")
                    ct = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
                    clip = comfy.sd.load_clip(ckpt_paths=[cp], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=ct)

        vae_files = resolved_files.get("vae", [])
        if vae_files:
            vp = _find_file_in_folders(vae_files[0], ["vae"])
            if not vp: raise ValueError(f"Bundle Loader: VAE '{vae_files[0]}' not found.")
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vp))

        log_node(f"Bundle Loader: ✅ {category}/{version} ready.", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": model_name},)


