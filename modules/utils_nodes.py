import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
from .common import UmeBundle, log_node
from .logger import logger


class UmeAiRT_Label:
    """A purely visual node used for organizing and annotating ComfyUI workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "title": ("STRING", {"default": "Label Title", "multiline": False, "tooltip": "Title text shown at the top of the label node."}),
                "text": ("STRING", {"default": "Description or Notes", "multiline": True, "tooltip": "Main text content of the label. Use for notes or workflow documentation."}),
                "color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"], {"default": "white", "tooltip": "Color of the label text (e.g. 'white', '#FF0000', 'cyan')."}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 100, "tooltip": "Text size in pixels. Default 16 is readable, increase for headers."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "do_label"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True

    def do_label(self, title, text, color, font_size):
        """Does nothing functionally — purely a frontend visual label."""
        return {}

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
                "files_bundle": ("UME_BUNDLE", {"tooltip": "Connect a Model Loader output here to see its individual model components."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, files_bundle):
        """Extracts model components from the UmeBundle dataclass."""
        return (
            files_bundle.model,
            files_bundle.clip,
            files_bundle.vae,
            files_bundle.model_name,
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

class UmeAiRT_Unpack_Prompt:
    """Deconstructs a UME_PROMPTS bundle into distinct Positive and Negative text strings."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompts": ("UME_PROMPTS",)}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"
    def unpack(self, prompts):
        """Extracts text prompts from the dictionary.

        Args:
            prompts (dict): The bundle containing "positive" and "negative" keys.

        Returns:
            tuple: A tuple containing (str(positive_prompt), str(negative_prompt)).
        """
        return (prompts.get("positive", ""), prompts.get("negative", ""))

class UmeAiRT_Log_Viewer:
    """A Node that retrieves and outputs the most recent logs from the UmeAiRT_Logger instance.

    Ideal for creating debugging panels directly inside ComfyUI interfaces without 
    relying exclusively on the background terminal window.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "refresh_trigger": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Change this value to re-fetch logs."}),
                "limit": ("INT", {"default": 20, "min": 1, "max": 100, "tooltip": "How many recent log entries to show (1-100)."}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_text",)
    FUNCTION = "get_logs"
    CATEGORY = "UmeAiRT/Utils"
    
    def get_logs(self, refresh_trigger, limit):
        """Pulls the log history to a single formatted string.

        Args:
            refresh_trigger (int): A pseudo-variable to force ComfyUI execution.
            limit (int): How many historical log lines to fetch.

        Returns:
            tuple: A tuple containing a single multiline string of recent logs.
        """
        logs = logger.get_logs(limit)
        text = "\n".join(logs)
        return (text,)


# --- Legacy Unpack Nodes Restoration ---

class UmeAiRT_Faces_Unpack_Node:
    """A legacy passthrough node ensuring old workflows using UME_FACES do not break."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "faces_bundle": ("UME_FACES", {"tooltip": "Connect a FaceDetailer output here to extract individual face crops."}),
            }
        }

    RETURN_TYPES = ("UME_FACES",)
    RETURN_NAMES = ("faces_passthrough",)
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, faces_bundle):
        """Passes the object forward.

        Args:
            faces_bundle (Any): The legacy faces data object.

        Returns:
            tuple: The untouched object.
        """
        return (faces_bundle,)

class UmeAiRT_Tags_Unpack_Node:
    """Legacy unpacking for UME_TAGS bundled data into a raw string."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tags_bundle": ("UME_TAGS", {"tooltip": "Connect a tagger output here to extract the individual tag strings."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_string",)
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, tags_bundle):
        """Forces the generic tags bundle into a string representation.

        Args:
            tags_bundle (Any): The legacy tags object.

        Returns:
            tuple: A tuple containing the string cast.
        """
        return (str(tags_bundle),)

class UmeAiRT_Pipe_Unpack_Node:
    """Legacy unpacking node mapping a monolithic UME_PIPE tuple/dict into standard outputs."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe_bundle": ("UME_PIPE", {"tooltip": "Connect a pipeline here to extract its components."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, pipe_bundle):
         """Analyzes the legacy pipeline format (list or dict) and extracts the 5 core variables.

         Args:
             pipe_bundle (dict|list): The incoming structured pipeline.

         Returns:
             tuple: A tuple mapping to (Model, Clip, VAE, Positive_text, Negative_text).

         Raises:
             ValueError: If the structure is completely unrecognized.
         """
         if not isinstance(pipe_bundle, dict):
             if isinstance(pipe_bundle, (list, tuple)) and len(pipe_bundle) >= 5:
                  return (pipe_bundle[0], pipe_bundle[1], pipe_bundle[2], pipe_bundle[3], pipe_bundle[4])
             # Safety fallback
             raise ValueError("UmeAiRT Unpack: Input is not a valid UME_PIPE bundle.")
         
         return (
             pipe_bundle.get("model"),
             pipe_bundle.get("clip"),
             pipe_bundle.get("vae"),
             pipe_bundle.get("positive_text", ""),
             pipe_bundle.get("negative_text", "")
         )


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
UmeAiRT_Unpack_PromptsBundle = UmeAiRT_Unpack_Prompt
UmeAiRT_Unpack_PipelineBundle = UmeAiRT_Unpack_Pipeline

class UmeAiRT_HealthCheck:
    """Startup node to validate dependencies and optimizations."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": True, "label_on": "Run", "label_off": "Skip", "tooltip": "Increment this number to re-run the health check."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Report",)
    FUNCTION = "run_check"
    CATEGORY = "UmeAiRT/Utils"

    def run_check(self, trigger):
        if not trigger:
            return ("Skipped",)
            
        from .optimization_utils import check_optimizations, get_cuda_memory
        import psutil
        
        report = []
        log_node("--- UmeAiRT Toolkit Health Check ---", color="CYAN")
        
        # 1. System Memory
        try:
            ram = psutil.virtual_memory()
            ram_report = f"RAM: {ram.available / (1024**3):.2f}GB / {ram.total / (1024**3):.2f}GB"
            log_node(ram_report, color="WHITE")
            report.append(ram_report)
        except Exception as e:
            err = f"RAM Check Failed: {e}"
            log_node(err, color="RED")
            report.append(err)
            
        # 2. VRAM
        try:
            vram_report = f"VRAM: {get_cuda_memory()}"
            log_node(vram_report, color="WHITE")
            report.append(vram_report)
        except Exception as e:
            err = f"VRAM Check Failed: {e}"
            log_node(err, color="RED")
            report.append(err)
            
        # 3. Optimizations
        try:
            opt_report = f"Optimizations: {check_optimizations()}"
            log_node(opt_report, color="WHITE")
            report.append(opt_report)
        except Exception as e:
            err = f"Opt Check Failed: {e}"
            log_node(err, color="RED")
            report.append(err)
            
        # 4. Bundles JSON
        try:
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umeairt_bundles.json")
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                b_report = "Bundles JSON: Valid"
            else:
                b_report = "Bundles JSON: Not Found"
            log_node(b_report, color="WHITE")
            report.append(b_report)
        except Exception as e:
            err = f"Bundles JSON: parsing failed ({e})"
            log_node(err, color="RED")
            report.append(err)
            
        log_node("------------------------------------", color="CYAN")
        
        return ("\n".join(report),)

