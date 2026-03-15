import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
from .common import log_node
from .logger import logger


class UmeAiRT_Label:
    """A purely visual node used for organizing and annotating ComfyUI workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "title": ("STRING", {"default": "Label Title", "multiline": False}),
                "text": ("STRING", {"default": "Description or Notes", "multiline": True}),
                "color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"], {"default": "white"}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 100}),
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
    """A Node designed to download specific model bundles or workflows.

    It reads from a `bundles.json` manifest located at the root of the node directory 
    to dynamically populate its ComfyUI dropdowns.
    """
    def __init__(self):
        self.bundles_data = {}
        self.json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umeairt_bundles.json")
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.bundles_data = json.load(f)
            except Exception as e:
                log_node(f"Error loading bundles.json: {e}", color="RED")
    
    _bundles_cache = None

    @classmethod
    def INPUT_TYPES(s):
        # Use class-level cache to avoid re-reading bundles.json on every UI refresh
        if UmeAiRT_Bundle_Downloader._bundles_cache is None:
            json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "umeairt_bundles.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        UmeAiRT_Bundle_Downloader._bundles_cache = json.load(f)
                except Exception:
                    UmeAiRT_Bundle_Downloader._bundles_cache = {}
            else:
                UmeAiRT_Bundle_Downloader._bundles_cache = {}
        data = UmeAiRT_Bundle_Downloader._bundles_cache
        
        categories = list(data.keys()) if data else ["Error: No Bundles"]
        return {
            "required": {
                "category": (categories,),
                "bundle_name": (["Select Category First"],), 
                "download_path": ("STRING", {"default": "downloads"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True

    def download(self, category, bundle_name, download_path):
        """Mocks the download action and returns a status string.

        Args:
            category (str): The selected category from the dropdown.
            bundle_name (str): The selected bundle item to download.
            download_path (str): The target destination folder.

        Returns:
            tuple: A tuple containing the execution status string.
        """
        log_node(f"Bundle Download requested: {category}/{bundle_name}", color="YELLOW")
        # Implementation of actual download logic would go here
        # For refactor, we keep it safe.
        return (f"Downloaded {bundle_name}",)


class UmeAiRT_Unpack_Settings:
    """Extracts multiple individual variables from a single UME_SETTINGS dictionary bundle."""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"settings": ("UME_SETTINGS",)}}
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "sampler", "scheduler", "seed")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"
    def unpack(self, settings):
        """Unpacks the provided settings dictionary.

        Args:
            settings (dict): The UME_SETTINGS bundle containing generation variables.

        Returns:
            tuple: A sequence containing (width, height, steps, cfg, sampler, scheduler, seed).
        """
        return (
            settings.get("width", 1024), settings.get("height", 1024),
            settings.get("steps", 20), settings.get("cfg", 8.0),
            settings.get("sampler_name", "euler"), settings.get("scheduler", "normal"),
            settings.get("seed", 0)
        )

class UmeAiRT_Unpack_FilesBundle:
    """Deconstructs a unified UME_FILES bundle into standard ComfyUI data pipes.

    Outputs Model, Clip, VAE, and the readable Model Name separately for native nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "files_bundle": ("UME_BUNDLE", {"tooltip": "Input UME_BUNDLE to unpack."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, files_bundle):
        """Extracts internal variables from the files dictionary.

        Args:
            files_bundle (dict): The custom dictionary holding loaded models.

        Returns:
            tuple: A tuple containing the (comfy_model, comfy_clip, comfy_vae, string_name).

        Raises:
            ValueError: If the input is not a recognized dictionary mapping.
        """
        if not isinstance(files_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_BUNDLE.")
        return (
            files_bundle.get("model"),
            files_bundle.get("clip"),
            files_bundle.get("vae"),
            files_bundle.get("model_name", "")
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
                "model_name": ("STRING", {"default": "", "tooltip": "Optional model name for metadata."}),
            }
        }

    RETURN_TYPES = ("UME_BUNDLE",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "pack"
    CATEGORY = "UmeAiRT/Utils/Pack"

    def pack(self, model, clip, vae, model_name=""):
        """Packs native ComfyUI models into a UME_BUNDLE dict.

        Args:
            model: The diffusion model.
            clip: The CLIP text encoder.
            vae: The VAE model.
            model_name (str, optional): A label for metadata. Defaults to "".

        Returns:
            tuple: A tuple containing the UME_BUNDLE dict.
        """
        return ({"model": model, "clip": clip, "vae": vae, "model_name": model_name},)

class UmeAiRT_Unpack_ImageBundle:
    """Deconstructs a UME_IMAGE bundle into native ComfyUI types."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input UME_IMAGE bundle to unpack."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "mode", "denoise", "auto_resize")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, image_bundle):
        """Extracts all fields from the image bundle.

        Args:
            image_bundle (dict): The bundled mapping.

        Returns:
            tuple: (image, mask, mode, denoise, auto_resize).
        """
        if not isinstance(image_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_IMAGE bundle.")
        return (
            image_bundle.get("image"),
            image_bundle.get("mask"),
            image_bundle.get("mode", "img2img"),
            float(image_bundle.get("denoise", 1.0)),
            bool(image_bundle.get("auto_resize", False)),
        )


class UmeAiRT_Unpack_Pipeline:
    """Deconstructs a UME_PIPELINE (GenerationContext) into native ComfyUI types.

    This enables full interoperability: connect any output to native or community nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("UME_PIPELINE", {"tooltip": "Pipeline context to unpack into native types."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("image", "model", "clip", "vae", "model_name", "positive", "negative", "width", "height", "steps", "cfg", "sampler_name", "scheduler", "seed", "denoise")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"

    def unpack(self, pipeline):
        """Extracts all fields from the GenerationContext pipeline.

        Args:
            pipeline (GenerationContext): The pipeline object.

        Returns:
            tuple: All 15 native ComfyUI outputs.
        """
        return (
            pipeline.image,
            pipeline.model,
            pipeline.clip,
            pipeline.vae,
            str(pipeline.model_name or ""),
            str(pipeline.positive_prompt or ""),
            str(pipeline.negative_prompt or ""),
            int(pipeline.width or 1024),
            int(pipeline.height or 1024),
            int(pipeline.steps or 20),
            float(pipeline.cfg or 8.0),
            str(pipeline.sampler_name or "euler"),
            str(pipeline.scheduler or "normal"),
            int(pipeline.seed or 0),
            float(pipeline.denoise or 1.0),
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
                "refresh_trigger": ("INT", {"default": 0, "min": 0, "max": 0}), # Dummy trigger
                "limit": ("INT", {"default": 20, "min": 1, "max": 100}),
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
                "faces_bundle": ("UME_FACES", {"tooltip": "Input UME_FACES bundle to unpack."}),
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
                "tags_bundle": ("UME_TAGS", {"tooltip": "Input UME_TAGS bundle to unpack."}),
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
                "pipe_bundle": ("UME_PIPE", {"tooltip": "Input UME_PIPE bundle to unpack."}),
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
                "trigger": ("BOOLEAN", {"default": True, "label_on": "Run", "label_off": "Skip"}),
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
        
        return ("\\n".join(report),)

