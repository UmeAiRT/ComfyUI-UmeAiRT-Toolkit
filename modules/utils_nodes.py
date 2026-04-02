import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
from .common import UmeBundle, log_node
from .logger import logger



class UmeAiRT_Bundle_Downloader:
    """Standalone model downloader utility node.

    Downloads model bundles from umeairt_bundles.json to the correct ComfyUI
    model folders WITHOUT loading them into memory. Ideal for:
    - Pre-downloading models on RunPod/cloud before running workflows
    - Batch-downloading entire model families (FLUX, Z-IMG)
    - Ensuring all required files are present before generation

    Uses aria2c for multi-connection downloads when available, with urllib fallback.
    Supports HuggingFace token for authenticated/faster downloads.
    """

    @classmethod
    def INPUT_TYPES(s):
        from .block_loaders import _get_bundle_dropdowns
        categories, versions_list = _get_bundle_dropdowns()
        return {
            "required": {
                "category": (categories, {"tooltip": "Model family to download (e.g. FLUX, Z-IMAGE_TURBO)."}),
                "version": (versions_list, {"tooltip": "Quantization/precision variant (e.g. fp16, GGUF_Q4)."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True

    def download(self, category, version):
        """Download all files for the selected bundle without loading into memory."""
        from .block_loaders import _download_bundle_files

        try:
            _, _, downloaded, skipped, errors = _download_bundle_files(category, version)
        except ValueError as e:
            return (f"❌ {e}",)

        parts = [f"📥 {category}/{version}:"]
        if downloaded:
            parts.append(f"{downloaded} downloaded")
        if skipped:
            parts.append(f"{skipped} already present")
        if errors:
            parts.append(f"{len(errors)} failed ({', '.join(errors)})")
        status = " | ".join(parts)
        log_node(status, color="GREEN" if not errors else "RED")
        return (status,)


class UmeAiRT_Unpack_Settings:
    """Extracts multiple individual variables from a single UME_SETTINGS dictionary bundle."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"settings": ("UME_SETTINGS",)}}
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "*", "*", "INT")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "sampler", "scheduler", "seed")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"
    def unpack(self, settings):
        """Unpacks the provided settings dataclass."""
        return (
            settings.width, settings.height,
            settings.steps, settings.cfg,
            settings.sampler_name, settings.scheduler,
            settings.seed
        )

class UmeAiRT_Unpack_FilesBundle:
    """Deconstructs a unified UME_FILES bundle into standard ComfyUI data pipes.

    Outputs Model, Clip, VAE, and the readable Model Name separately for native nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Connect a Model Loader output here to see its individual model components."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, model_bundle):
        """Extracts model components from the UmeBundle dataclass."""
        return (
            model_bundle.model,
            model_bundle.clip,
            model_bundle.vae,
            model_bundle.model_name,
        )


class UmeAiRT_Pack_Bundle:
    """Packs native ComfyUI types (MODEL, CLIP, VAE) into a UME_BUNDLE.

    Use this to feed models from any native or community loader into the Block pipeline.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP text encoder."}),
                "vae": ("VAE", {"tooltip": "The VAE model."}),
            },
            "optional": {
                "model_name": ("STRING", {"default": "", "tooltip": "Model name stored in the bundle metadata (useful for Image Saver)."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "pack"
    CATEGORY = "UmeAiRT/Utils/Pack"

    def pack(self, model, clip, vae, model_name=""):
        """Packs native ComfyUI models into a UmeBundle."""
        return (UmeBundle(model=model, clip=clip, vae=vae, model_name=model_name),)

class UmeAiRT_Unpack_ImageBundle:
    """Deconstructs a UME_IMAGE bundle into native ComfyUI types."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Connect an Image process output here to see its individual components."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "mode", "denoise", "auto_resize")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, image_bundle):
        """Extracts all fields from the UmeImage dataclass."""
        return (
            image_bundle.image,
            image_bundle.mask,
            image_bundle.mode,
            float(image_bundle.denoise),
            bool(image_bundle.auto_resize),
        )


class UmeAiRT_Unpack_Pipeline:
    """Deconstructs a UME_PIPELINE (GenerationContext) into native ComfyUI types.

    This enables full interoperability: connect any output to native or community nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("UME_PIPELINE", {"tooltip": "Connect a generation pipeline to extract all its values as individual outputs."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "*", "*", "INT", "FLOAT")
    RETURN_NAMES = ("image", "model", "clip", "vae", "model_name", "positive", "negative", "width", "height", "steps", "cfg", "sampler_name", "scheduler", "seed", "denoise")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, gen_pipe):
        """Extracts all fields from the GenerationContext pipeline.

        Args:
            pipeline (GenerationContext): The pipeline object.

        Returns:
            tuple: All 15 native ComfyUI outputs.
        """
        return (
            gen_pipe.image,
            gen_pipe.model,
            gen_pipe.clip,
            gen_pipe.vae,
            str(gen_pipe.model_name or ""),
            str(gen_pipe.positive_prompt or ""),
            str(gen_pipe.negative_prompt or ""),
            int(gen_pipe.width or 1024),
            int(gen_pipe.height or 1024),
            int(gen_pipe.steps or 20),
            float(gen_pipe.cfg or 8.0),
            str(gen_pipe.sampler_name or "euler"),
            str(gen_pipe.scheduler or "normal"),
            int(gen_pipe.seed or 0),
            float(gen_pipe.denoise or 1.0),
        )





# --- Legacy Unpack Nodes Restoration ---


class UmeAiRT_Signature:
    """A Node designed purely for aesthetic and branding purposes on the canvas.

    It renders a custom transparent PNG signature (`assets/signature.png`) via JavaScript.
    It has no inputs, and running the node yields an empty result.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}, # No inputs! Clean and simple.
        }

    RETURN_TYPES = ()
    FUNCTION = "display_signature"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True 

    def display_signature(self):
        """Silently short-circuits execution context.

        Returns:
            dict: An empty UI images dictionary since rendering is entirely handled client-side.
        """
        # The node execution does nothing except return the path relative to ComfyUI for preview.
        # But this node is for frontend visual mostly!
        # If the user somehow executes it, we just return empty.
        # The real magic happens in umeairt_signature.js
        return {"ui": {"images": []}}

# Aliases for legacy compatibility
UmeAiRT_Unpack_SettingsBundle = UmeAiRT_Unpack_Settings
UmeAiRT_Unpack_PipelineBundle = UmeAiRT_Unpack_Pipeline


