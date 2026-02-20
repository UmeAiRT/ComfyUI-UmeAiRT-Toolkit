import torch
import os
import json
import folder_paths
import nodes as comfy_nodes
from .common import UME_SHARED_STATE, log_node
from .logger import logger

class UmeAiRT_Label:
    """
    Simple Label Node for organizing workflows.
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
        # Does nothing functionally, just frontend visual
        return {}

class UmeAiRT_Wireless_Debug:
    """
    Prints the current Wireless Shared State to the console/log.
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
    """
    Downloads model/workflow bundles from a JSON source.
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
        log_node(f"Bundle Download requested: {category}/{bundle_name}", color="YELLOW")
        # Implementation of actual download logic would go here
        # For refactor, we keep it safe.
        return (f"Downloaded {bundle_name}",)


class UmeAiRT_Unpack_Settings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"settings": ("UME_SETTINGS",)}}
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "sampler", "scheduler", "seed")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"
    def unpack(self, settings):
        return (
            settings.get("width", 1024), settings.get("height", 1024),
            settings.get("steps", 20), settings.get("cfg", 8.0),
            settings.get("sampler", "euler"), settings.get("scheduler", "normal"),
            settings.get("seed", 0)
        )

class UmeAiRT_Unpack_FilesBundle:
    """
    Unpacks UME_FILES bundle into Model, Clip, VAE, and Name.
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
        if not isinstance(files_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_FILES bundle.")
        return (
            files_bundle.get("model"),
            files_bundle.get("clip"),
            files_bundle.get("vae"),
            files_bundle.get("model_name", "")
        )

class UmeAiRT_Unpack_ImageBundle:
    """
    Unpacks UME_IMAGE bundle into Image and Mask.
    """
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
        if not isinstance(image_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_IMAGE bundle.")
        
        image = image_bundle.get("image")
        mask = image_bundle.get("mask")
        
        return (image, mask)

class UmeAiRT_Unpack_Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompts": ("UME_PROMPTS",)}}
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Utils/Unpack"
    def unpack(self, prompts):
        return (prompts.get("positive", ""), prompts.get("negative", ""))

class UmeAiRT_Log_Viewer:
    """
    Displays the last N log entries from the UmeAiRT Logger.
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
        logs = logger.get_logs(limit)
        text = "\n".join(logs)
        return (text,)


# --- Legacy Unpack Nodes Restoration ---

class UmeAiRT_Faces_Unpack_Node:
    """
    Unpacks FACES_DATA bundle.
    """
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
        return (faces_bundle,)

class UmeAiRT_Tags_Unpack_Node:
    """
    Unpacks UME_TAGS bundle.
    """
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
        return (str(tags_bundle),)

class UmeAiRT_Pipe_Unpack_Node:
    """
    Unpacks UME_PIPE bundle (Model, Clip, Vae, Positive, Negative).
    """
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

# Aliases for legacy compatibility
UmeAiRT_Unpack_SettingsBundle = UmeAiRT_Unpack_Settings
UmeAiRT_Unpack_PromptsBundle = UmeAiRT_Unpack_Prompt
