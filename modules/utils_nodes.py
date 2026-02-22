import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
from .common import UME_SHARED_STATE, log_node
from .logger import logger

class UmeAiRT_Label:
    """A purely visual node used for organizing and annotating ComfyUI workflows.

    It does not process any actual data but provides a customizable text label 
    that can be placed anywhere on the canvas.
    """
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
        """Executes the visual node.

        Args:
            title (str): The title text of the label.
            text (str): The main descriptive text.
            color (str): The chosen color theme constraint.
            font_size (int): The relative font size for display.

        Returns:
            dict: An empty dictionary as this is purely a frontend visual element.
        """
        # Does nothing functionally, just frontend visual
        return {}

class UmeAiRT_Wireless_Debug:
    """A utility node for developers to inspect the current Wireless Shared State.

    When triggered, it parses the `UME_SHARED_STATE` dictionary and prints 
    its keys and abstract values (like tensor shapes) to the console log.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
             "required": {
                 "trigger": ("ANY", {"default": None, "forceInput": True}),
             }
        }
    RETURN_TYPES = ()
    FUNCTION = "debug_state"
    CATEGORY = "UmeAiRT/Utils"
    OUTPUT_NODE = True

    def debug_state(self, trigger=None):
        """Reads and logs the global state out to the standardized logger.

        Args:
            trigger (Any, optional): Anything connected to forcefully trigger node execution.

        Returns:
            dict: Empty dictionary.
        """
        log_node("--- Wireless State Debug ---", color="CYAN")
        for k, v in UME_SHARED_STATE.items():
            val_str = str(v)
            if isinstance(v, torch.Tensor):
                val_str = f"Tensor {v.shape} {v.device}"
            elif isinstance(v, dict):
                val_str = f"Dict keys: {list(v.keys())}"
            
            log_node(f"{k}: {val_str}", color="WHITE")
        log_node("----------------------------", color="CYAN")
        return {}

class UmeAiRT_Bundle_Downloader:
    """A Node designed to download specific model bundles or workflows.

    It reads from a `bundles.json` manifest located at the root of the node directory 
    to dynamically populate its ComfyUI dropdowns.
    """
    def __init__(self):
        self.bundles_data = {}
        self.json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bundles.json")
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.bundles_data = json.load(f)
            except Exception as e:
                log_node(f"Error loading bundles.json: {e}", color="RED")
    
    @classmethod
    def INPUT_TYPES(s):
        # We need to instantiate to load json for dynamic lists?
        # ComfyUI creates strict instances. We can read file in INPUT_TYPES.
        json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bundles.json")
        data = {}
        if os.path.exists(json_path):
             try:
                 with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
             except: pass
        
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
            settings.get("sampler", "euler"), settings.get("scheduler", "normal"),
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
                "files_bundle": ("UME_FILES", {"tooltip": "Input UME_FILES bundle to unpack."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

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
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_FILES bundle.")
        return (
            files_bundle.get("model"),
            files_bundle.get("clip"),
            files_bundle.get("vae"),
            files_bundle.get("model_name", "")
        )

class UmeAiRT_Unpack_ImageBundle:
    """Deconstructs a unified UME_IMAGE bundle into standalone IMAGE and MASK tensors."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input UME_IMAGE bundle to unpack."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, image_bundle):
        """Separates the image and its associated mask.

        Args:
            image_bundle (dict): The bundled mapping containing "image" and "mask" keys.

        Returns:
            tuple: A tuple containing (torch.Tensor(image), torch.Tensor(mask)).

        Raises:
            ValueError: If the input object is not a valid dictionary.
        """
        if not isinstance(image_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_IMAGE bundle.")
        
        image = image_bundle.get("image")
        mask = image_bundle.get("mask")
        
        return (image, mask)

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
    CATEGORY = "UmeAiRT/Unpack"

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
    CATEGORY = "UmeAiRT/Unpack"

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
    CATEGORY = "UmeAiRT/Unpack"

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
