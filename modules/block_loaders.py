"""ComfyUI node definitions for model loading (Files Settings and Bundle Loader).

This module defines the UI-facing ComfyUI nodes. Download logic is in
download_utils.py and manifest handling is in manifest.py.
"""
import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, UmeBundle, log_node
from typing import Tuple, Dict, Any

# Re-export from refactored modules for backward compatibility
from .download_utils import get_hf_token, download_file, verify_file_hash
from .manifest import (
    PATH_TYPE_TO_FOLDERS, find_file_in_folders, get_download_dest,
    load_manifest, get_bundle_dropdowns, download_bundle_files,
)

# Legacy aliases (used internally by some existing code)
_get_hf_token = get_hf_token
_find_file_in_folders = find_file_in_folders
_get_download_dest = get_download_dest
_PATH_TYPE_TO_FOLDERS = PATH_TYPE_TO_FOLDERS
_get_bundle_dropdowns = get_bundle_dropdowns
_download_bundle_files = download_bundle_files
_download_file = download_file


# --- Shared Loader Helper ---

def _load_diffusion_model(filename, folder="diffusion_models"):
    """Load a diffusion model (safetensors or GGUF) with dtype auto-detection.

    Centralizes the duplicated GGUF/dtype logic shared by FLUX, ZIMG, and Bundle loaders.

    Args:
        filename (str): The model filename (may end in .gguf or .safetensors).
        folder (str): The ComfyUI folder category (default: "diffusion_models").

    Returns:
        The loaded diffusion model object.
    """
    if filename.endswith(".gguf"):
        from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
        return UnetLoaderGGUF().load_unet(filename)[0]
    model_path = folder_paths.get_full_path(folder, filename)
    model_options = {}
    ln = filename.lower()
    if "e4m3fn" in ln:
        model_options["dtype"] = torch.float8_e4m3fn
    elif "e5m2" in ln:
        model_options["dtype"] = torch.float8_e5m2
    return comfy.sd.load_diffusion_model(model_path, model_options=model_options)


# --- Files / Model Loaders (Block) ---

class UmeAiRT_FilesSettings_Checkpoint:
    """Simplified loader for standard SD1.5 / SDXL checkpoints."""

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        vaes = ["Baked VAE"] + folder_paths.get_filename_list("vae")
        return {
            "required": {
                "ckpt_name": (checkpoints, {"tooltip": "Select a checkpoint file."}),
            },
            "optional": {
                "vae_name": (vaes, {"default": "Baked VAE", "advanced": True, "tooltip": "Select an external VAE or use the one baked into the checkpoint."}),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1, "advanced": True, "tooltip": "Stop at which CLIP layer. -1 is default. -2 is common for anime models."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "UmeAiRT/Block/Loaders"
    OUTPUT_NODE = True

    def load_checkpoint(self, ckpt_name, vae_name="Baked VAE", clip_skip=-1):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)
        model, clip, vae_ckpt = out[:3]

        if clip_skip < -1:
            clip = clip.clone()
            clip.clip_layer(clip_skip)

        if vae_name != "Baked VAE":
            vae_path = folder_paths.get_full_path("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        else:
            vae = vae_ckpt
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=ckpt_name),)







class UmeAiRT_FilesSettings_FLUX:
    """FLUX model loader (diffusion model + dual CLIP + VAE).

    Loads a FLUX architecture model with its required dual text encoder
    (CLIP-L + T5-XXL) and VAE. Supports both standard safetensors and
    quantized GGUF model formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models = folder_paths.get_filename_list("diffusion_models")
        clips = folder_paths.get_filename_list("clip")
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The FLUX diffusion model file (e.g. flux1-dev-fp8.safetensors or a GGUF variant)."}),
                "clip_1": (clips, {"tooltip": "First text encoder — typically CLIP-L (clip_l.safetensors)."}),
                "clip_2": (clips, {"tooltip": "Second text encoder — typically T5-XXL (t5xxl_fp16.safetensors or GGUF)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. ae.safetensors for FLUX)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_flux"
    CATEGORY = "UmeAiRT/Block/Loaders"
    OUTPUT_NODE = True

    def load_flux(self, diff_model, clip_1, clip_2, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # Dual CLIP
        if clip_1.endswith(".gguf") or clip_2.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import DualCLIPLoaderGGUF
            clip = DualCLIPLoaderGGUF().load_clip(clip_1, clip_2, type="flux")[0]
        else:
            clip_paths = [
                folder_paths.get_full_path("clip", clip_1),
                folder_paths.get_full_path("clip", clip_2),
            ]
            clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        # VAE
        vae_path = folder_paths.get_full_path("vae", vae)
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (UmeBundle(model=model, clip=clip, vae=vae_obj, model_name=model_name),)


class UmeAiRT_FilesSettings_Fragmented:
    """Fragmented Model Loader for multi-file model architectures.

    Loads models split across multiple safetensors files using ComfyUI's
    native DiffusersLoader. Designed for models distributed as folders
    (e.g. HuggingFace Diffusers format).
    """

    @classmethod
    def INPUT_TYPES(s):
        paths = []
        for search_path in folder_paths.get_folder_paths("diffusion_models"):
            if os.path.isdir(search_path):
                for f in os.listdir(search_path):
                    full_path = os.path.join(search_path, f)
                    if os.path.isdir(full_path):
                        paths.append(f)
        return {
            "required": {
                "model_path": (paths, {"tooltip": "Select a model folder from your diffusion_models directory. Folders typically contain index.json + sharded safetensors files."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_diffusers"
    CATEGORY = "UmeAiRT/Block/Loaders"
    OUTPUT_NODE = True

    def load_diffusers(self, model_path):
        model_name = model_path
        node = comfy_nodes.DiffusersLoader()
        model, clip, vae = node.load_checkpoint(model_path)[:3]
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=model_name),)


class UmeAiRT_FilesSettings_ZIMG:
    """Z-IMAGE model loader (Lumina2 architecture).

    Loads a Z-IMAGE architecture model with its Qwen text encoder
    and VAE. Supports safetensors and GGUF quantized formats.
    """

    @classmethod
    def INPUT_TYPES(s):
        diff_models = folder_paths.get_filename_list("diffusion_models")
        clips = folder_paths.get_filename_list("clip")
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "diff_model": (diff_models, {"tooltip": "The Z-IMAGE diffusion model file (e.g. z-image-turbo-bf16.safetensors or GGUF)."}),
                "clip": (clips, {"tooltip": "Qwen text encoder (e.g. qwen3-4b.safetensors or GGUF)."}),
                "vae": (vaes, {"tooltip": "VAE model (e.g. ae.safetensors)."}),
            }
        }
    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_zimg"
    CATEGORY = "UmeAiRT/Block/Loaders"
    OUTPUT_NODE = True

    def load_zimg(self, diff_model, clip, vae):
        model_name = diff_model
        # Model
        model = _load_diffusion_model(diff_model)
        # CLIP (Lumina2 / Qwen)
        if clip.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip_obj = CLIPLoaderGGUF().load_clip(clip, type="lumina2")[0]
        else:
            clip_path = folder_paths.get_full_path("clip", clip)
            ct = getattr(comfy.sd.CLIPType, "LUMINA2", comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_obj = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=ct
            )
        # VAE
        vae_path = folder_paths.get_full_path("vae", vae)
        vae_obj = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (UmeBundle(model=model, clip=clip_obj, vae=vae_obj, model_name=model_name),)


class UmeAiRT_BundleLoader:
    """Bundle Auto-Loader: select a model + version, auto-download missing files, and load them.

    Combines downloading and loading into one node.
    Reads from the remote model manifest to populate dropdowns and determine loading strategy.
    """

    @classmethod
    def INPUT_TYPES(s):
        categories, versions_list = get_bundle_dropdowns()
        return {
            "required": {
                "category": (categories, {"tooltip": "Select model family (e.g. FLUX/Dev, Z-IMAGE/Turbo)."}),
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
        resolved_files, meta, downloaded, skipped, errors = download_bundle_files(category, version)
        if errors:
            raise RuntimeError(f"Bundle Loader: {len(errors)} file(s) failed to download: {', '.join(errors)}")
        loader_type = meta.get("loader_type", "zimg")
        clip_type_str = meta.get("clip_type", "lumina2")

        model, clip, vae = None, None, None
        model_name = ""
        model_pt = None
        for pt_key in ["zimg_diff", "flux_diff", "wan_diff", "hidream_diff", "qwen_diff",
                        "ltxv_diff", "ltx2_diff", "zimg_unet", "flux_unet"]:
            if pt_key in resolved_files: model_pt = pt_key; break
        if model_pt:
            model_filename = resolved_files[model_pt][0]
            model_name = model_filename
            if model_filename.endswith(".gguf"):
                from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
                model = UnetLoaderGGUF().load_unet(model_filename)[0]
            else:
                model_path = find_file_in_folders(model_filename, PATH_TYPE_TO_FOLDERS.get(model_pt, ["diffusion_models"]))
                if not model_path: raise ValueError(f"Bundle Loader: Model '{model_filename}' not found.")
                model_options = {}
                ln = model_filename.lower()
                if "e4m3fn" in ln: model_options["dtype"] = torch.float8_e4m3fn
                elif "e5m2" in ln: model_options["dtype"] = torch.float8_e5m2
                model = comfy.sd.load_diffusion_model(model_path, model_options=model_options)

        # Collect all text encoder files from various path types
        clip_files = []
        for te_key in ["clip", "text_encoders_t5", "text_encoders_qwen",
                        "text_encoders_gemma", "text_encoders_llama", "text_encoders_ltx"]:
            clip_files.extend(resolved_files.get(te_key, []))
        if clip_files:
            if loader_type == "flux" and len(clip_files) >= 2:
                clip_paths = [find_file_in_folders(cf, ["clip", "text_encoders"]) for cf in clip_files]
                ct = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.FLUX)
                clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=ct)
            else:
                cf = clip_files[0]
                if cf.endswith(".gguf"):
                    from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
                    clip = CLIPLoaderGGUF().load_clip(cf, type=clip_type_str)[0]
                else:
                    cp = find_file_in_folders(cf, ["clip", "text_encoders"])
                    if not cp: raise ValueError(f"Bundle Loader: CLIP '{cf}' not found.")
                    ct = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
                    clip = comfy.sd.load_clip(ckpt_paths=[cp], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=ct)

        vae_files = resolved_files.get("vae", [])
        if vae_files:
            vp = find_file_in_folders(vae_files[0], ["vae"])
            if not vp: raise ValueError(f"Bundle Loader: VAE '{vae_files[0]}' not found.")
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vp))

        log_node(f"Bundle Loader: ✅ {category}/{version} ready.", color="GREEN")
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=model_name),)
