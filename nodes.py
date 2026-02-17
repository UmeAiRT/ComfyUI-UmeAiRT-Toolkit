"""
UmeAiRT Toolkit - Wireless Node System for ComfyUI
--------------------------------------------------
This toolkit provides a "wireless" workflow experience by storing logic state
(Models, VAE, CLIP, Parameters) in a global shared dictionary.
Nodes act as "Setters" (Input) or "Getters" (Output/Processor).

Author: UmeAiRT Team
License: MIT
"""

import os
import sys
import math
import random
import re
import numpy as np
from PIL import Image

import torch
import folder_paths
import nodes as comfy_nodes
import comfy.samplers
import comfy.utils
from server import PromptServer

from server import PromptServer

# --- UmeAiRT Logger ---
try:
    from .logger import log_node, CYAN, GREEN, RED, RESET
except ImportError:
    # Fallback if relative import fails (e.g. running script directly)
    # But usually unnecessary in ComfyUI context
    from logger import log_node, CYAN, GREEN, RED, RESET



# Global Storage
UME_SHARED_STATE = {}


# Internal Keys
KEY_IMAGESIZE = "ume_internal_imagesize"
KEY_FPS = "ume_internal_fps"
KEY_STEPS = "ume_internal_steps"
KEY_DENOISE = "ume_internal_denoise"
KEY_SEED = "ume_internal_seed"
KEY_SCHEDULER = "ume_internal_scheduler"
KEY_SAMPLER = "ume_internal_sampler"
KEY_CFG = "ume_internal_cfg"
KEY_POSITIVE = "ume_internal_positive"
KEY_NEGATIVE = "ume_internal_negative"
KEY_MODEL = "ume_internal_model"
KEY_VAE = "ume_internal_vae"
KEY_CLIP = "ume_internal_clip"
KEY_LATENT = "ume_internal_latent"
KEY_MODEL_NAME = "ume_internal_model_name"
KEY_LORAS = "ume_internal_loras"
KEY_SOURCE_IMAGE = "ume_internal_source_image"
KEY_SOURCE_MASK = "ume_internal_source_mask"
KEY_CONTROLNETS = "ume_internal_controlnets"

# --- HELPER FUNCTIONS ---

def resize_tensor(tensor, target_h, target_w, interp_mode="bilinear", is_mask=False):
    """
    Helper function to resize image or mask tensors.
    Handles dimension permutations for ComfyUI (B,H,W,C) -> Torch (B,C,H,W) -> Resize -> Back.
    """
    if is_mask:
        # Mask: [B, H, W] -> [B, 1, H, W]
        t = tensor.unsqueeze(1)
    else:
        # Image: [B, H, W, C] -> [B, C, H, W]
        t = tensor.permute(0, 3, 1, 2)
    
    t_resized = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode=interp_mode, align_corners=False if interp_mode!="nearest" else None)
    
    if is_mask:
        # [B, 1, H, W] -> [B, H, W]
        return t_resized.squeeze(1)
    else:
        # [B, C, H, W] -> [B, H, W, C]
        return t_resized.permute(0, 2, 3, 1)


# --- GUIDANCE NODES ---

class UmeAiRT_Guidance_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guidance": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_value"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_value(self, guidance):
        UME_SHARED_STATE[KEY_CFG] = guidance
        return ()

class UmeAiRT_Guidance_Output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {} 
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cfg",)
    FUNCTION = "get_value"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_value(self):
        val = UME_SHARED_STATE.get(KEY_CFG, 8.0)
        return (float(val),)


# --- IMAGE SIZE NODES ---

class UmeAiRT_ImageSize_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_size"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_size(self, width, height):
        UME_SHARED_STATE[KEY_IMAGESIZE] = {"width": width, "height": height}
        return ()

class UmeAiRT_ImageSize_Output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_size(self):
        val = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
        return (int(val.get("width", 1024)), int(val.get("height", 1024)))


# --- FPS NODES ---

class UmeAiRT_FPS_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, fps):
        UME_SHARED_STATE[KEY_FPS] = fps
        return ()

class UmeAiRT_FPS_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("fps",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_FPS, 24)
        return (int(val),)


# --- STEPS NODES ---

class UmeAiRT_Steps_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "display": "slider"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, steps):
        UME_SHARED_STATE[KEY_STEPS] = steps
        return ()

class UmeAiRT_Steps_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_STEPS, 20)
        return (int(val),)


# --- DENOISE NODES ---

class UmeAiRT_Denoise_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, denoise):
        UME_SHARED_STATE[KEY_DENOISE] = denoise
        return ()

class UmeAiRT_Denoise_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("denoise",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_DENOISE, 1.0)
        return (float(val),)


# --- SEED NODES ---

class UmeAiRT_Seed_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, seed):
        UME_SHARED_STATE[KEY_SEED] = seed
        return ()

class UmeAiRT_Seed_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_SEED, 0)
        return (int(val),)


# --- SCHEDULER NODES ---

class UmeAiRT_Scheduler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, scheduler):
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()

class UmeAiRT_Scheduler_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        # Default to first scheduler if missing
        default = comfy.samplers.KSampler.SCHEDULERS[0] if comfy.samplers.KSampler.SCHEDULERS else "normal"
        val = UME_SHARED_STATE.get(KEY_SCHEDULER, default)
        return (val,)


# --- SAMPLER NODES ---

class UmeAiRT_Sampler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, sampler_name):
        UME_SHARED_STATE[KEY_SAMPLER] = sampler_name
        return ()

class UmeAiRT_Sampler_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        # Default to first sampler if missing
        default = comfy.samplers.KSampler.SAMPLERS[0] if comfy.samplers.KSampler.SAMPLERS else "euler"
        val = UME_SHARED_STATE.get(KEY_SAMPLER, default)
        return (val,)


class UmeAiRT_SamplerScheduler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, sampler, scheduler):
        UME_SHARED_STATE[KEY_SAMPLER] = sampler
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()


# --- POSITIVE PROMPT NODES ---

class UmeAiRT_Positive_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, text):
        UME_SHARED_STATE[KEY_POSITIVE] = text
        return ()

class UmeAiRT_Positive_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_POSITIVE, "")
        return (str(val),)


# --- NEGATIVE PROMPT NODES ---

class UmeAiRT_Negative_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "text, watermark", "multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, text):
        UME_SHARED_STATE[KEY_NEGATIVE] = text
        return ()

class UmeAiRT_Negative_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_NEGATIVE, "")
        return (str(val),)


# --- MODEL/VAE/CLIP NODES ---

class UmeAiRT_Model_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, model):
        UME_SHARED_STATE[KEY_MODEL] = model
        return ()

class UmeAiRT_Model_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_MODEL)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless MODEL set!")
        return (val,)

class UmeAiRT_VAE_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, vae):
        UME_SHARED_STATE[KEY_VAE] = vae
        return ()

class UmeAiRT_VAE_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_VAE)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless VAE set!")
        return (val,)

class UmeAiRT_CLIP_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, clip):
        UME_SHARED_STATE[KEY_CLIP] = clip
        return ()

class UmeAiRT_CLIP_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_CLIP)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless CLIP set!")
        return (val,)


# --- CHECKPOINT NODES ---

class UmeAiRT_WirelessCheckpointLoader(comfy_nodes.CheckpointLoaderSimple):
    """
    Sets the global Wireless Model, CLIP, VAE, and Model Name.
    Resets the Wireless LoRA list to empty on each load to ensure fresh state.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint_wireless"
    CATEGORY = "UmeAiRT/Wireless/Loaders"
    OUTPUT_NODE = True 
    
    def load_checkpoint_wireless(self, ckpt_name):
        # Call original loader
        out = super().load_checkpoint(ckpt_name)
        # Set Wireless State
        UME_SHARED_STATE[KEY_MODEL] = out[0]
        UME_SHARED_STATE[KEY_CLIP] = out[1]
        UME_SHARED_STATE[KEY_VAE] = out[2]
        UME_SHARED_STATE[KEY_MODEL_NAME] = ckpt_name
        UME_SHARED_STATE[KEY_LORAS] = []
        log_node(f"Wireless Checkpoint Loaded: {ckpt_name}", color="GREEN")
        return out


# --- IMAGE NODES (Source Input) ---

class UmeAiRT_WirelessImageLoader(comfy_nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "mode": ("BOOLEAN", {"default": False, "label_on": "Inpaint", "label_off": "Img2Img"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "load_image_wireless"
    CATEGORY = "UmeAiRT/Wireless/Loaders"
    OUTPUT_NODE = True

    def load_image_wireless(self, image, resize, mode):
        # Call original loader
        out = super().load_image(image)
        img = out[0]
        mask = out[1]

        # Mode Logic: If Img2Img (False), we discard mask for global state
        # But we still process resizing if needed
        
        if resize:
            # Fetch Target Size
            size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
            target_w = int(size.get("width", 1024))
            target_h = int(size.get("height", 1024))
            
            # --- Result Holders ---
            final_img = img
            final_mask = mask

            # --- Helper: Resize & Crop ---
            def resize_and_crop(tensor, mode="bilinear", is_mask=False):
                # tensor shape: [B, H, W, C] (Image) or [B, H, W] (Mask)
                
                # 1. Permute to [B, C, H, W] for interpolate
                if is_mask:
                    # Mask is [B, H, W] -> Add C dim -> [B, 1, H, W]
                    t = tensor.unsqueeze(1)
                else:
                     # Image is [B, H, W, C] -> [B, C, H, W]
                    t = tensor.permute(0, 3, 1, 2)

                b, c, h, w = t.shape
                
                # 2. Scale (Aspect Ratio Preserved - Cover)
                scale = max(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize
                if mode == "bilinear":
                    t_resized = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode=mode, align_corners=False)
                else:
                    t_resized = torch.nn.functional.interpolate(t, size=(new_h, new_w), mode=mode)
                
                # 3. Center Crop
                # Calculate start indices
                h_start = (new_h - target_h) // 2
                w_start = (new_w - target_w) // 2
                
                if h_start < 0: h_start = 0 
                if w_start < 0: w_start = 0
                
                t_cropped = t_resized[:, :, h_start:h_start+target_h, w_start:w_start+target_w]
                
                # 4. Restore Shape
                if is_mask:
                    # [B, 1, H, W] -> [B, H, W]
                    return t_cropped.squeeze(1)
                else:
                    # [B, C, H, W] -> [B, H, W, C]
                    return t_cropped.permute(0, 2, 3, 1)

            # Apply Logic
            if img is not None:
                final_img = resize_and_crop(img, mode="bilinear", is_mask=False)
            
            if mask is not None:
                final_mask = resize_and_crop(mask, mode="nearest", is_mask=True)
            
            # Update References
            img = final_img
            mask = final_mask
            
            # print(f"UmeAiRT Wireless Image Loader: Resized to {target_w}x{target_h}")

        # Set Wireless State
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        log_node(f"Wireless Image Loaded: {image} (Resize: {resize}, Mode: {'Inpaint' if mode else 'Img2Img'})", color="GREEN")
        
        # Apply Mode (Inpaint vs Img2Img)
        if mode:
            # Inpaint: Store Mask
            UME_SHARED_STATE[KEY_SOURCE_MASK] = mask
        else:
            # Img2Img: Clear/Nullify Mask
            UME_SHARED_STATE[KEY_SOURCE_MASK] = None

        return ()

class UmeAiRT_SourceImage_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        img = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)
        if img is None:
            raise ValueError("UmeAiRT: No Wireless Source Image available!")
        if mask is None:
             pass
        return (img, mask)


# --- LATENT NODES ---

class UmeAiRT_Latent_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, latent):
        UME_SHARED_STATE[KEY_LATENT] = latent
        return ()

class UmeAiRT_Latent_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_LATENT)
        if val is None:
             # Return empty dummy if missing to avoid crash, but KSampler checks validation naturally
             raise ValueError("UmeAiRT: No Wireless Latent set.")
        return (val,)


# --- WIRELESS KSAMPLER ---

class UmeAiRT_WirelessKSampler:
    """
    Autonomous KSampler that retrieves all inputs (Model, VAE, CLIP, Latent, Params)
    from the global wireless state.
    
    Features:
    - Auto-Mode Switching:
      - If denoise >= 1.0: Runs in Txt2Img mode (Generates empty latent).
      - If denoise < 1.0: Runs in Img2Img mode.
        - Automatically fetches 'Wireless Source Image'.
        - Encodes it using the wireless VAE.
        - Uses result as input latent.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Auto-detects mode based on KEY_LATENT presence
            },
            "optional": {
                "signal": ("*",), # Generic input for syncing
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Wireless/Samplers"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always run if upstream variables changed (which we can't track easily without linking).
        # But since we access globals that fluctuate, we return NaN to force update.
        return float("nan")

    def __init__(self):
        self.cnet_loader = comfy_nodes.ControlNetLoader()
        self.cnet_apply = comfy_nodes.ControlNetApplyAdvanced()

    def process(self, signal=None):
        # 0. Check Wireless Image Bundle for ControlNets (Implicit Linear support)
        # Note: Wireless KSampler normally consumes global state. 
        # But for ControlNets to work here from the linear workflow, 
        # the user must have used 'ControlNet Image Apply' which outputs a bundle.
        # The 'Wireless Image Process' updates global KEY_SOURCE_IMAGE but does NOT store the bundle stack globally.
        # So... pure wireless sampling won't see these ControlNets unless we updated global state with them.
        # BUT: The user asked to remove the 'controlnets' input. 
        # This implies they might only use this with 'Block Sampler' OR they expect the bundle data to be available globally?
        # WAIT: If they use 'Wireless KSampler', they usually don't pipe anything into it except Signal.
        # If they rely on 'ControlNet Image Apply', that node outputs a bundle.
        # If that bundle isn't connected to KSampler, KSampler can't see it.
        # 'Wireless Image Process' does NOT output 'UME_IMAGE' bundle by default unless we change it?
        # Actually `ControlNet Image Apply` takes `UME_IMAGE` and outputs `UME_IMAGE`.
        # If the user uses `Wireless KSampler`, it has no `image` input.
        # So `Wireless KSampler` effectively LOSES ControlNet support with this removal, UNLESS:
        # 1. We start storing `controlnets` stack in `UME_SHARED_STATE`.
        # 2. `ControlNet Image Apply` is updated to write to `UME_SHARED_STATE`.
        
        # Assumption: I will update `ControlNet Image Apply` to write to global state in the next step.
        # So here I will read from Global State.
        
        controlnets = UME_SHARED_STATE.get(KEY_CONTROLNETS, [])
        
        # 1. Fetch Objects
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        if model is None:
            raise ValueError("❌ Wireless KSampler: No MODEL found!\n\n💡 Solution: Add a 'Wireless Checkpoint Loader' or 'Model Loader (Block)' node to load your model.")
        if vae is None:
            raise ValueError("❌ Wireless KSampler: No VAE found!\n\n💡 Solution: Add a 'Wireless Checkpoint Loader' or 'Model Loader (Block)' node (VAE is included with checkpoints).")
        if clip is None:
            raise ValueError("❌ Wireless KSampler: No CLIP found!\n\n💡 Solution: Add a 'Wireless Checkpoint Loader' or 'Model Loader (Block)' node (CLIP is included with checkpoints).")

        # 2. Fetch Params
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
        cfg = float(UME_SHARED_STATE.get(KEY_CFG, 8.0))
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        denoise = float(UME_SHARED_STATE.get(KEY_DENOISE, 1.0))
        
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        log_node(f"Starting Wireless Sampling: {steps} steps | CFG: {cfg} | Denoise: {denoise} | Sampler: {sampler_name}", color="MAGENTA")

        # 3. Handle Latent (Auto-Detect Txt2Img vs Img2Img)
        wireless_latent = UME_SHARED_STATE.get(KEY_LATENT)
        latent_image = None
        
        # Check for Img2Img via Denoise
        if denoise < 1.0:
            # Try to fetch source image for Img2Img
            source_image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
            source_mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)

            if source_image is not None:
                # print(f"UmeAiRT DEBUG: Wireless KSampler triggered Img2Img (Denoise {denoise}). Encoding Source Image.")
                log_node(f"Wireless KSampler: Img2Img Triggered (Denoise {denoise:.2f}). Encoding Source Image...", color="YELLOW")
                # VAE Encode Logic
                # VAEEncode needs (pixels, vae) -> returns (latent,)
                try:
                    latent_image = comfy_nodes.VAEEncode().encode(vae, source_image)[0]
                    
                    # Apply Mask for Inpainting if present AND not empty
                    # LoadImage returns a mask of 0s for opaque images. 
                    # If we apply that as a noise_mask, KSampler changes nothing (keeps original).
                    if source_mask is not None:
                        if torch.any(source_mask > 0):
                            # ComfyUI 'SetLatentNoiseMask' node logic: s["noise_mask"] = mask
                            latent_image["noise_mask"] = source_mask
                            log_node("Wireless KSampler: Applying Inpainting Mask.", color="YELLOW")
                        else:
                            # Mask is all zeros (Opaque image default). Ignore it to allow full Img2Img.
                            pass

                except Exception as e:
                     log_node(f"Error: Failed to VAE Encode source image: {e}", color="RED")
                     # Fallback to Txt2Img if encode fails? better to raise error or fallback to empty.
                     pass
        
        # Fallback / Txt2Img Check
        if latent_image is None:
            if wireless_latent is not None:
                 # Explicit Latent Input overrides
                 latent_image = wireless_latent
            else:
                # Txt2Img Mode (Create Empty Latent)
                size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
                width = int(size.get("width", 1024))
                height = int(size.get("height", 1024))
                
                # Create Logic copied from EmptyLatentImage
                batch_size = 1
                latent = torch.zeros([batch_size, 4, height // 8, width // 8])
                latent_image = {"samples": latent}

        # 4. Encode Prompts
        # Positive
        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]

        # 4.5 Apply ControlNets (Wireless Injection)
        if controlnets:
            # print(f"UmeAiRT Wireless KSampler: Applying {len(controlnets)} ControlNets...")
            log_node(f"Wireless KSampler: Applying {len(controlnets)} ControlNets...", color="MAGENTA")
            for cnet_def in controlnets:
                c_name, c_image, c_str, c_start, c_end = cnet_def
                if c_name != "None" and c_image is not None:
                    try:
                         # print(f"  - ControlNet: {c_name} | Str: {c_str}")
                         cnet_model = self.cnet_loader.load_controlnet(c_name)[0]
                         # ControlNetApplyAdvanced takes both pos and neg
                         positive, negative = self.cnet_apply.apply_controlnet(positive, negative, cnet_model, c_image, c_str, c_start, c_end)
                    except Exception as e:
                        log_node(f"Failed to apply {c_name}: {e}", color="RED")

        # 5. Sample
        # We reuse the standard KSampler function
        return comfy_nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)


class UmeAiRT_WirelessInpaintComposite:
    """
    Composites the generated image with the source image using the Inpaint Mask.
    Auto-fetches 'Wireless Source Image' and 'Wireless Source Mask'.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def composite(self, image):
        # 1. Fetch Wireless Inputs
        source_image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        source_mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)

        if source_image is None:
             # print("UmeAiRT Composite Warning: No Wireless Source Image found. Returning generated image.")
             log_node("Composite Warning: No Wireless Source Image found. Returning generated image.", color="RED")
             return (image,)

        
        if source_mask is None:
             # print("UmeAiRT Composite Warning: No Wireless Source Mask found. Returning generated image.")
             log_node("Composite Warning: No Wireless Source Mask found. Returning generated image.", color="RED")
             return (image,)


        # 2. Logic: Similar to ImageCompositeMasked
        # Resize source_image and mask to match generated image [B, H, W, C]
        B, H, W, C = image.shape
        
        # --- Source Image Resize ---
        source_resized = source_image
        # Check if source needs resize (B, H, W, C)
        sB, sH, sW, sC = source_resized.shape
        if sH != H or sW != W:
             # Permute to [B, C, H, W] for interpolate
             s_p = source_resized.permute(0, 3, 1, 2)
             s_resized = torch.nn.functional.interpolate(s_p, size=(H, W), mode="bilinear", align_corners=False)
             source_resized = s_resized.permute(0, 2, 3, 1)

        # --- Mask Resize ---
        mask_resized = source_mask
        # Mask can be [H, W] or [B, H, W]
        if len(mask_resized.shape) == 2:
            mask_resized = mask_resized.unsqueeze(0) # [1, H, W]
        
        mB, mH, mW = mask_resized.shape
        if mH != H or mW != W:
            # Add channel dim for interpolate: [B, 1, H, W]
            m_p = mask_resized.unsqueeze(1)
            m_resized = torch.nn.functional.interpolate(m_p, size=(H, W), mode="bilinear", align_corners=False)
            mask_resized = m_resized.squeeze(1)

        # --- Composite ---
        # source * (1 - mask) + dest * mask
        
        # Ensure mask has [B, H, W, 1] shape
        m = mask_resized
        # If [B, H, W], unsqueeze last
        if len(m.shape) == 3:
            m = m.unsqueeze(-1)
        elif len(m.shape) == 2:
            m = m.unsqueeze(0).unsqueeze(-1)
        
        # Repeat batch if needed
        if m.shape[0] < B:
            m = m.repeat(B, 1, 1, 1)
        
        # Ensure source batch match
        if source_resized.shape[0] < B:
            source_resized = source_resized.repeat(B, 1, 1, 1)    

        # Clamp mask
        m = torch.clamp(m, 0.0, 1.0)
        
        # Composite
        # Areas with Mask=1 -> Generated Image (Inpainted Area)
        # Areas with Mask=0 -> Source Image (Original Area)
        
        result = source_resized * (1.0 - m) + image * m
        
        return (result,)


class UmeAiRT_Label:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                " ": ("STRING", {"default": "TITLE", "multiline": False}),
                "  ": ("STRING", {"default": "Description...", "multiline": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "UmeAiRT/Utilities"

    def do_nothing(self, **kwargs):
        return ()


# --- WIRELESS DEBUG NODES ---

class UmeAiRT_Wireless_Debug:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug_info",)
    FUNCTION = "debug_state"
    CATEGORY = "UmeAiRT/Utilities"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always re-run to show latest state
        return float("nan")

    def debug_state(self):
        # Fetch all
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        seed = UME_SHARED_STATE.get(KEY_SEED, "Unset")
        steps = UME_SHARED_STATE.get(KEY_STEPS, "Unset")
        cfg = UME_SHARED_STATE.get(KEY_CFG, "Unset")
        sampler = UME_SHARED_STATE.get(KEY_SAMPLER, "Unset")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "Unset")
        denoise = UME_SHARED_STATE.get(KEY_DENOISE, "Unset")
        size = UME_SHARED_STATE.get(KEY_IMAGESIZE, "Unset")
        latent = UME_SHARED_STATE.get(KEY_LATENT)
        pos = UME_SHARED_STATE.get(KEY_POSITIVE, "Unset")
        neg = UME_SHARED_STATE.get(KEY_NEGATIVE, "Unset")

        if isinstance(size, dict):
            size_str = f"{size.get('width', '?')}x{size.get('height', '?')}"
        else:
            size_str = str(size)

        # Format
        info = f"""--- UmeAiRT Wireless State ---
Model: {'Loaded' if model else 'MISSING'}
VAE: {'Loaded' if vae else 'MISSING'}
CLIP: {'Loaded' if clip else 'MISSING'}
Latent: {'Loaded' if latent is not None else 'MISSING (Txt2Img only)'}

Seed: {seed}
Steps: {steps}
CFG: {cfg}
Sampler: {sampler}
Scheduler: {scheduler}
Denoise: {denoise}
Size: {size_str}

Positive: {pos[:50]}...
Negative: {neg[:50]}...
------------------------------"""
        
        # print(info) # Also print to console
        log_node("Wireless State Debug:", color="CYAN")
        print(info)

        
        # Return text to UI and as output
        return {"ui": {"text": (info,)}, "result": (info,)}


# --- LOADER NODES ---

class UmeAiRT_MultiLoraLoader:
    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                
                # SLOT 1
                "lora_1": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "lora_1_name": (lora_list,),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                
                # SLOT 2
                "lora_2": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "lora_2_name": (lora_list,),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                
                # SLOT 3
                "lora_3": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "lora_3_name": (lora_list,),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "UmeAiRT/Loaders"

    def load_loras(self, model, clip, 
                   lora_1, lora_1_name, lora_1_strength,
                   lora_2, lora_2_name, lora_2_strength,
                   lora_3, lora_3_name, lora_3_strength):
        
        # Helper to apply lora if enabled
        loaded_loras = UME_SHARED_STATE.get(KEY_LORAS, [])
        if not isinstance(loaded_loras, list): loaded_loras = []

        def apply_lora(curr_model, curr_clip, is_on, name, strength):
            if is_on and name != "None":
                # LoraLoader returns (model, clip)
                # Helper to update key
                loaded_loras.append({"name": name, "strength": strength})
                return self.lora_loader.load_lora(curr_model, curr_clip, name, strength, strength)
            return curr_model, curr_clip
        
        # Pipeline
        m, c = apply_lora(model, clip, lora_1, lora_1_name, lora_1_strength)
        m, c = apply_lora(m, c, lora_2, lora_2_name, lora_2_strength)
        m, c = apply_lora(m, c, lora_3, lora_3_name, lora_3_strength)
        
        UME_SHARED_STATE[KEY_LORAS] = loaded_loras

        return (m, c)


# --- WRAPPER NODES (EXTERNAL INTEGRATIONS) ---

class UmeAiRT_WirelessUltimateUpscale_Base:
    """Base class to handle common USDU imports and logic."""
    def get_usdu_node(self):
        # 1. Imports (Local Library)
        usdu_path = os.path.join(os.path.dirname(__file__), "usdu_core")
        if usdu_path not in sys.path:
            sys.path.append(usdu_path)
        try:
            import usdu_main 
            return usdu_main.UltimateSDUpscale()
        except ImportError as e:
             raise ImportError(f"UmeAiRT: Could not import local UltimateSDUpscale files from {usdu_path}. Error: {e}")

    def fetch_wireless_common(self):
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))
        
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        
        # Generator params (fetched for fallback or base calc)
        gen_steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20)) 
        
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        
        cfg = float(UME_SHARED_STATE.get(KEY_CFG, 8.0))
        denoise = float(UME_SHARED_STATE.get(KEY_DENOISE, 1.0))
        
        size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 512, "height": 512})
        tile_width = int(size.get("width", 512))
        tile_height = int(size.get("height", 512))

        if not model or not vae or not clip:
            missing = []
            if not model: missing.append("MODEL")
            if not vae: missing.append("VAE")
            if not clip: missing.append("CLIP")
            raise ValueError(f"❌ Wireless Upscale: Missing {', '.join(missing)}!\n\n💡 Solution: Add a 'Wireless Checkpoint Loader' or 'Model Loader (Block)' node to load your model.")
            
        return model, vae, clip, pos_text, neg_text, seed, gen_steps, sampler_name, scheduler, tile_width, tile_height, cfg, denoise

    def encode_prompts(self, clip, pos_text, neg_text):
         # Positive
        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]
        
        # Negative
        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]
        
        return positive, negative


class UmeAiRT_WirelessUltimateUpscale(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Global Prompt"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def upscale(self, image, enabled, model, upscale_by, clean_prompt=True):
        # print(f"DEBUG: UmeAiRT Wireless Upscale Simple - Enabled: {enabled}")
        log_node(f"Wireless Upscale Simple - Enabled: {enabled}", color="CYAN")

        if not enabled:
            return (image,)

        # Hardcoded recommended denoise for simple mode
        denoise = 0.35

        # Load Upscale Model Internally
        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
             raise ImportError("UmeAiRT: Could not import UpscaleModelLoader from comfy_extras. Ensure ComfyUI is standard.")
             
        usdu_node = self.get_usdu_node()
        model, vae, clip, pos_text, neg_text, seed, gen_steps, sampler_name, scheduler, t_w, t_h, wireless_cfg, wireless_denoise = self.fetch_wireless_common()
        
        # Clean Prompt Logic
        if clean_prompt:
            # print("UmeAiRT Upscale: Using Clean Prompt (Empty Positive) to prevent hallucinations.")
            log_node("Upscale: Using Clean Prompt (Empty Positive) to prevent hallucinations.", color="YELLOW")

            # Use empty positive prompt
            target_pos_text = ""
        else:
            target_pos_text = pos_text
            
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)
        
        # Logic: 1/4 steps, rounded up
        steps = math.ceil(gen_steps / 4)
        
        # Settings
        cfg = 1.0 # Force 1.0 for texture enhancing
        # denoise is passed as arg
        mode_type = "Linear"
        mask_blur = 16
        tile_padding = 32
        seam_fix_mode = "None"
        seam_fix_denoise = 1.0
        
        # Force uniform tiles to prevent artifacts
        force_uniform = True

        return usdu_node.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=t_w, tile_height=t_h, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
            force_uniform_tiles=force_uniform, tiled_decode=False,
            suppress_preview=True,
        )


class UmeAiRT_WirelessUltimateUpscale_Advanced(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                
                # Advanced Settings
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Global Prompt"}),
                
                "mode_type": (["Linear", "Chess", "None"], {"default": "Linear"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                
                "seam_fix_mode": (["None", "Band Pass", "Half Tile", "Half Tile + Intersections"], {"default": "None"}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 8}),
                
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_advanced"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def upscale_advanced(self, image, enabled, model, upscale_by, denoise, clean_prompt, 
                         mode_type, tile_width, tile_height, mask_blur, tile_padding,
                         seam_fix_mode, seam_fix_denoise, seam_fix_width, seam_fix_mask_blur, seam_fix_padding,
                         force_uniform_tiles, tiled_decode):
        
        if not enabled:
            return (image,)

        # Load Upscale Model
        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
             raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")
             
        usdu_node = self.get_usdu_node()
        model, vae, clip, pos_text, neg_text, seed, gen_steps, sampler_name, scheduler, _, _, wireless_cfg, wireless_denoise = self.fetch_wireless_common()
        
        # Clean Prompt
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)
        
        # Logic
        steps = math.ceil(gen_steps / 4)
        cfg = 1.0 # Force 1.0

        return usdu_node.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=tile_width, tile_height=tile_height, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=seam_fix_mask_blur, seam_fix_width=seam_fix_width, seam_fix_padding=seam_fix_padding,
            force_uniform_tiles=force_uniform_tiles, tiled_decode=tiled_decode,
            suppress_preview=True,
        )


class UmeAiRT_WirelessFaceDetailer_Advanced(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "enabled": ("BOOLEAN", {"default": True}),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": 4096, "step": 1, "default": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Generation"

    def face_detail(self, image, bbox_detector, enabled, guide_size, guide_size_for, max_size,
                    denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size):
        
        if not enabled:
            return (image,)

        # Fetch Wireless State
        model, vae, clip, pos_text, neg_text, wireless_seed, wireless_steps, wireless_sampler, wireless_scheduler, _, _, wireless_cfg, wireless_denoise = self.fetch_wireless_common()
        positive, negative = self.encode_prompts(clip, pos_text, neg_text)

        # Import Logic Locally (Lazy Import)
        try:
            from .facedetailer_core import logic as fd_logic
        except ImportError:
            # Fallback for direct import if package structure differs
            import facedetailer_core.logic as fd_logic

        # Detect Faces
        # We assume bbox_detector follows the standard detect interface: detect(image, threshold, dilation, crop_factor, drop_size)
        # Returns (shape, [seg1, seg2...])
        # print(f"DEBUG: bbox_detector object: {bbox_detector}")
        # print(f"DEBUG: bbox_detector type: {type(bbox_detector)}")
        # print(f"DEBUG: bbox_detector dir: {dir(bbox_detector)}")
        
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

        log_node(f"Face Detailer (Adv): Detected {len(segs)} faces/regions. Processing...", color="MAGENTA")

        # Run Detailer with Wireless Settings
        # Usage strategy: We prioritize wireless settings as requested by user.
        # NOTE: If wireless_denoise is 1.0 (default), it might be too high for detailing, but we respect the user's intent to use setters.
        
        result = fd_logic.do_detail(
            image=image, segs=segs, model=model, clip=clip, vae=vae,
            guide_size=guide_size, guide_size_for_bbox=guide_size_for, max_size=max_size,
            seed=wireless_seed, steps=wireless_steps, cfg=wireless_cfg, sampler_name=wireless_sampler, scheduler=wireless_scheduler,
            positive=positive, negative=negative, denoise=denoise,
            feather=feather, noise_mask=noise_mask, force_inpaint=force_inpaint,
            drop_size=drop_size
        )
        
        return result



        return result


class UmeAiRT_BboxDetectorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("bbox"),),
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR",)
    FUNCTION = "load_bbox"
    CATEGORY = "UmeAiRT/Loaders"

    def load_bbox(self, model_name):
        try:
            from .facedetailer_core import detector
        except ImportError:
            import facedetailer_core.detector as detector
        
        return (detector.load_bbox_model(model_name),)


class UmeAiRT_WirelessFaceDetailer_Simple(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "model": (folder_paths.get_filename_list("bbox"),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "face_detail_simple"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def face_detail_simple(self, image, enabled, model, denoise):
        
        if not enabled:
            return (image,)

        # Defaults (Hardcoded best practices)
        guide_size = 512
        guide_size_for_bbox = True
        max_size = 1024
        # denoise = 0.5 # Now passed as argument
        feather = 5
        noise_mask = True
        force_inpaint = True
        bbox_threshold = 0.5
        bbox_dilation = 10
        bbox_crop_factor = 3.0
        drop_size = 10

        # Fetch Wireless State
        sd_model, vae, clip, pos_text, neg_text, wireless_seed, wireless_steps, wireless_sampler, wireless_scheduler, _, _, wireless_cfg, _ = self.fetch_wireless_common()
        positive, negative = self.encode_prompts(clip, pos_text, neg_text)

        # Import Logic & Detector
        try:
            from .facedetailer_core import logic as fd_logic
            from .facedetailer_core import detector
        except ImportError:
            import facedetailer_core.logic as fd_logic
            import facedetailer_core.detector as detector

        # Load/Run Detector internally
        bbox_detector = detector.load_bbox_model(model)
        
        # Detect
        # print(f"DEBUG: Running Simple Detector with {bbox_model_name}")
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)
        
        log_node(f"Face Detailer (Simple): Detected {len(segs)} faces/regions. Processing...", color="MAGENTA")

        # Run Detailer
        result = fd_logic.do_detail(
            image=image, segs=segs, model=sd_model, clip=clip, vae=vae,
            guide_size=guide_size, guide_size_for_bbox=guide_size_for_bbox, max_size=max_size,
            seed=wireless_seed, steps=wireless_steps, cfg=wireless_cfg, sampler_name=wireless_sampler, scheduler=wireless_scheduler,
            positive=positive, negative=negative, denoise=denoise,
            feather=feather, noise_mask=noise_mask, force_inpaint=force_inpaint,
            drop_size=drop_size
        )
        
        return result



class UmeAiRT_WirelessImageSaver(UmeAiRT_WirelessUltimateUpscale_Base):
    """
    Autonomous Image Saver.
    - Resolves 'filename' pattern using wireless state (Date, Time, Seed, Model Name).
    - Fetches metadata (Hashes, Generation Params) from wireless state.
    - Saves image to output directory.
    - Provides preview to UI.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": 'SDXL/%date/%time_%basemodelname_%seed', "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "UmeAiRT/Wireless/Output"

    def save_images(self, images, filename, prompt=None, extra_pnginfo=None):
        
        # Split 'filename' input into path and filename
        # Ensure separators are standardized
        full_pattern = filename.replace("\\", "/")
        if "/" in full_pattern:
             path, filename = full_pattern.rsplit("/", 1)
        else:
             path = ""
             filename = full_pattern
             
        # Sanitize Filename (Manager Request)
        # Remove invalid characters: < > : " / \ | ? *
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
             
        # Import Logic Locally
        try:
            from .image_saver_core.logic import ImageSaverLogic
        except ImportError:
            import image_saver_core.logic as ImageSaverLogic

        # Hardcoded Defaults for Simple Mode
        extension = "png"
        lossless_webp = True
        quality_jpeg_or_webp = 100
        optimize_png = False
        embed_workflow = True
        save_workflow_as_json = False
        
        # Fetch shared state
        width = UME_SHARED_STATE.get(KEY_IMAGESIZE, {}).get("width", 512)
        height = UME_SHARED_STATE.get(KEY_IMAGESIZE, {}).get("height", 512)
        
        modelname = UME_SHARED_STATE.get(KEY_MODEL_NAME, "UmeAiRT_Wireless_Model")
        
        # Process LoRAs
        additional_hashes = ""
        loras = UME_SHARED_STATE.get(KEY_LORAS, [])
        if loras:
            try:
                from .image_saver_core.utils import full_lora_path_for, get_sha256
                hash_list = []
                for lora in loras:
                    name = lora.get("name")
                    strength = lora.get("strength", 1.0)
                    if name:
                        path_to_lora = full_lora_path_for(name)
                        if path_to_lora:
                            l_hash = get_sha256(path_to_lora)[:10]
                            # Format: name:hash:strength
                            hash_list.append(f"{name}:{l_hash}:{strength}")

                if hash_list:
                    additional_hashes = ",".join(hash_list)
            except Exception as e:
                # print(f"UmeAiRT Error processing LoRAs: {e}")
                log_node(f"Error processing LoRAs: {e}", color="RED")


        metadata_obj = ImageSaverLogic.make_metadata(
            modelname=modelname,
            positive=UME_SHARED_STATE.get(KEY_POSITIVE, ""),
            negative=UME_SHARED_STATE.get(KEY_NEGATIVE, ""),
            width=int(width),
            height=int(height),
            seed_value=int(UME_SHARED_STATE.get(KEY_SEED, 0)),
            steps=int(UME_SHARED_STATE.get(KEY_STEPS, 20)),
            cfg=float(UME_SHARED_STATE.get(KEY_CFG, 8.0)),
            sampler_name=UME_SHARED_STATE.get(KEY_SAMPLER, "euler"),
            scheduler_name=UME_SHARED_STATE.get(KEY_SCHEDULER, "normal"),
            denoise=float(UME_SHARED_STATE.get(KEY_DENOISE, 1.0)),
            clip_skip=0, 
            custom="UmeAiRT Wireless",
            additional_hashes=additional_hashes,
            download_civitai_data=False, 
            easy_remix=True
        )

        # 4. Resolve Path and Call Save
        time_format = "%Y-%m-%d-%H%M%S"
        resolved_path = ImageSaverLogic.replace_placeholders(
            path, 
            metadata_obj.width, metadata_obj.height, metadata_obj.seed, metadata_obj.modelname, 
            0, time_format, 
            metadata_obj.sampler_name, metadata_obj.steps, metadata_obj.cfg, metadata_obj.scheduler_name, 
            metadata_obj.denoise, metadata_obj.clip_skip, metadata_obj.custom
        )

        result_paths = ImageSaverLogic.save_images(
            images=images,
            filename_pattern=filename,
            extension=extension,
            path=resolved_path,
            quality_jpeg_or_webp=quality_jpeg_or_webp,
            lossless_webp=lossless_webp,
            optimize_png=optimize_png,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
            save_workflow_as_json=save_workflow_as_json,
            embed_workflow=embed_workflow,
            counter=0,
            time_format=time_format, 
            metadata=metadata_obj
        )

        log_node(f"Image Saver: Saved {len(result_paths)} images to {resolved_path}", color="GREEN")

        # 5. Format Output
        # subfolder = os.path.normpath(path) # Old logic used raw path
        
        ui_result = {
            "ui": {"images": [{"filename": os.path.basename(f), "subfolder": os.path.join(folder_paths.output_directory, resolved_path) if resolved_path else '', "type": 'output'} for f in result_paths]},
        }
        
        # Override to be cleaner (ImageSaver logic returns just filename usually)
        ui_result["ui"]["images"] = []
        for ckpt in result_paths:
             ui_result["ui"]["images"].append({
                 "filename": os.path.basename(ckpt),
                 "subfolder": resolved_path,
                 "type": "output"
             })
             
        return ui_result

class UmeAiRT_WirelessUltimateUpscale_Advanced(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        usdu_seam = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "mode_type": (usdu_modes, {"default": "Linear"}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 64}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
                "seam_fix_mode": (usdu_seam, {"default": "None"}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def upscale(self, image, upscale_model, upscale_by, steps, cfg, denoise, mode_type, mask_blur, tile_padding, seam_fix_mode, seam_fix_denoise):
        usdu_node = self.get_usdu_node()
        model, vae, clip, pos_text, neg_text, seed, _, sampler_name, scheduler, t_w, t_h = self.fetch_wireless_common()
        positive, negative = self.encode_prompts(clip, pos_text, neg_text)

        return usdu_node.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=t_w, tile_height=t_h, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
            force_uniform_tiles=True, tiled_decode=False
        )




# --- BLOCK NODES (Wireless + Bundle) ---

class UmeAiRT_GenerationSettings:
    """
    Bundles Generation Settings (Width, Height, Sampler, Schedule, Steps, CFG, Denoise, Seed).
    Updates Global Wireless State.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider"}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1, "display": "slider"}),
                "guidance": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5, "display": "slider"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("UME_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/Generation"
    OUTPUT_NODE = True

    def process(self, width, height, sampler, scheduler, steps, guidance, seed):
        # 1. Update Global State (Wireless Synergy)
        UME_SHARED_STATE[KEY_IMAGESIZE] = {"width": width, "height": height}
        UME_SHARED_STATE[KEY_SAMPLER] = sampler
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        UME_SHARED_STATE[KEY_STEPS] = steps
        UME_SHARED_STATE[KEY_CFG] = guidance
        UME_SHARED_STATE[KEY_SEED] = seed

        # 2. Return Bundle
        settings = {
            "width": width,
            "height": height,
            "sampler": sampler,
            "scheduler": scheduler,
            "steps": steps,
            "cfg": guidance,
            "seed": seed
        }
        return (settings,)


class UmeAiRT_FilesSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "vae_name": (["Baked"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks/Models"
    OUTPUT_NODE = True

class UmeAiRT_FilesSettings_Checkpoint:
    """
    Simple version - loads Model, CLIP, and VAE from Checkpoint.
    Uses baked VAE and default CLIP settings.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks/Models"

    def load_files(self, ckpt_name):
        # 1. Load Checkpoint
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        clip = out[1]
        vae = out[2]

        # 2. Update Global State
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = ckpt_name
        UME_SHARED_STATE[KEY_LORAS] = []

        # 3. Return Bundle
        files = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "model_name": ckpt_name
        }
        return (files,)


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
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                 "vae_name": (["Baked"] + folder_paths.get_filename_list("vae"),),
                 "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks"

    def load_files(self, ckpt_name, vae_name="Baked", clip_skip=-1):
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

        # 4. Update Global State
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = ckpt_name
        UME_SHARED_STATE[KEY_LORAS] = [] # Reset LoRAs on new checkpoint load

        # 5. Return Bundle
        files = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "model_name": ckpt_name
        }
        return (files,)


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
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }

    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks/Models"

    def load_files(self, unet_name, weight_dtype, clip_name1, clip_name2, vae_name):
        # 1. Load UNET
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_unet(unet_path)
        
        # 2. Load CLIPs (Dual Clip Loader Logic)
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # 3. Load VAE
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 4. Update Global State
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = unet_name
        UME_SHARED_STATE[KEY_LORAS] = [] # Reset LoRAs on new checkpoint load

        # 5. Return Bundle
        files = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "model_name": unet_name
        }
        log_node(f"Block Checkpoint (FLUX) Loaded: {unet_name}", color="GREEN")
        return (files,)


# Helper for generating LoRA inputs
def get_lora_inputs(count):
    inputs = {
        "required": {},
        "optional": {
            "loras": ("UME_LORA_STACK",),
        }
    }
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    for i in range(1, count + 1):
        inputs["optional"][f"lora_{i}_name"] = (lora_list,)
        inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"})
    return inputs

def process_lora_stack(loras, **kwargs):
    current_stack = []
    if loras:
        current_stack.extend(loras)
    
    # Iterate through kwargs to find lora definitions
    # We assume keys like lora_1_name, lora_1_strength
    
    # Extract indices
    indices = set()
    for k in kwargs.keys():
        if k.startswith("lora_") and "_name" in k:
            parts = k.split("_")
            # lora_{i}_name -> index is parts[1]
            if len(parts) >= 3 and parts[1].isdigit():
                indices.add(int(parts[1]))
    
    sorted_indices = sorted(list(indices))

    for i in sorted_indices:
        name = kwargs.get(f"lora_{i}_name")
        strength = kwargs.get(f"lora_{i}_strength", 1.0)
        
        if name and name != "None":
            # Unified strength for model and clip
            current_stack.append((name, strength, strength))
            
    return (current_stack,)

class UmeAiRT_LoraBlock_1:
    @classmethod
    def INPUT_TYPES(s):
        return get_lora_inputs(1)

    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"

    def process(self, loras=None, **kwargs):
        return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_3:
    @classmethod
    def INPUT_TYPES(s):
        return get_lora_inputs(3)

    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"

    def process(self, loras=None, **kwargs):
        return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_5:
    @classmethod
    def INPUT_TYPES(s):
        return get_lora_inputs(5)

    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"

    def process(self, loras=None, **kwargs):
        return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_10:
    @classmethod
    def INPUT_TYPES(s):
        return get_lora_inputs(10)

    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"

    def process(self, loras=None, **kwargs):
        return process_lora_stack(loras, **kwargs)





class UmeAiRT_ControlNetImageApply_Advanced:
    """
    Advanced Linear ControlNet Node.
    Full control over Start/End percent.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "optional_control_image": ("IMAGE",), 
            }
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "UmeAiRT/Blocks/ControlNet"

    def apply_controlnet(self, image_bundle, control_net_name, strength, start_percent, end_percent, optional_control_image=None):
        if not isinstance(image_bundle, dict):
            raise ValueError("ControlNet Image Apply: Input is not a valid UME_IMAGE bundle.")

        new_bundle = image_bundle.copy()
        cnet_stack = new_bundle.get("controlnets", [])
        if not isinstance(cnet_stack, list): cnet_stack = []
        
        if control_net_name != "None":
            control_use_image = optional_control_image if optional_control_image is not None else new_bundle.get("image")
            
            if control_use_image is None:
                raise ValueError("ControlNet Image Apply: No Image found in bundle and no optional image provided.")
                
            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent))
            
        new_bundle["controlnets"] = cnet_stack
        
        # Update Global State for Wireless KSampler support
        UME_SHARED_STATE[KEY_CONTROLNETS] = cnet_stack

        return (new_bundle,)



class UmeAiRT_ControlNetImageProcess:
    """
    Unified Node: Image Process + ControlNet Apply (Simple).
    Resizes image, sets denoise (txt2img/img2img), and applies ControlNet.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                # Image Process Params
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mode": (["img2img", "txt2img"], {"default": "img2img"}),
                
                # ControlNet Params
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
            },
            "optional": {
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
            }
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/ControlNet"

    def process(self, image_bundle, denoise, mode, control_net_name, strength, resize=False):
        if not isinstance(image_bundle, dict):
            raise ValueError("ControlNet Image Process: Input is not a valid UME_IMAGE bundle.")

        # --- 1. Image Processing Logic (Simplified) ---
        
        # Unpack
        image = image_bundle.get("image")
        mask = image_bundle.get("mask") # Can be None in bundle

        if image is None:
             raise ValueError("ControlNet Image Process: Bundle has no image.")

        # TXT2IMG Logic
        if mode == "txt2img":
             # print("UmeAiRT Unified CNet: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).")
             log_node("Unified CNet: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             denoise = 1.0

             mask = None 

        # RESIZE
        final_image = image
        final_mask = mask
        
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))

             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear", is_mask=False)
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)

        # Mode Handling for Bundle
        final_mode = "img2img"
        if mode == "txt2img":
             final_mode = "txt2img"
             final_mask = None
        elif mode == "img2img":
             final_mask = None # For Bundle (mask is hidden effectively for linear img2img unless used in inpaint)

        # Create New Bundle Base
        new_bundle = {
            "image": final_image,
            "mask": final_mask,
            "mode": final_mode,
            "denoise": denoise,
            "controlnets": image_bundle.get("controlnets", []).copy() if image_bundle.get("controlnets") else []
        }

        # --- 2. ControlNet Apply Logic (Simple) ---
        
        cnet_stack = new_bundle["controlnets"]
        
        if control_net_name != "None":
            # Always use the PROCESSED image as the control image
            control_use_image = final_image
            
            # Fixed Start/End
            start_percent = 0.0
            end_percent = 1.0
            
            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent))
            
        new_bundle["controlnets"] = cnet_stack
        
        # Update Global State for Wireless KSampler support
        UME_SHARED_STATE[KEY_CONTROLNETS] = cnet_stack

        return (new_bundle,)


class UmeAiRT_ControlNetImageApply_Simple:
    """
    Simple Linear ControlNet Node.
    Strength 0.0 - 2.0. Start=0.0, End=1.0 hardcoded.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "UmeAiRT/Blocks/ControlNet"

    def apply_controlnet(self, image_bundle, control_net_name, strength):
        if not isinstance(image_bundle, dict):
            raise ValueError("ControlNet Image Apply: Input is not a valid UME_IMAGE bundle.")

        new_bundle = image_bundle.copy()
        cnet_stack = new_bundle.get("controlnets", [])
        if not isinstance(cnet_stack, list): cnet_stack = []
        
        if control_net_name != "None":
            control_use_image = new_bundle.get("image")
            
            if control_use_image is None:
                raise ValueError("ControlNet Image Apply: No Image found in bundle.")
                
            # Fixed Start/End
            start_percent = 0.0
            end_percent = 1.0
            
            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent))
            
        new_bundle["controlnets"] = cnet_stack
        
        # Update Global State for Wireless KSampler support
        UME_SHARED_STATE[KEY_CONTROLNETS] = cnet_stack

        return (new_bundle,)


class UmeAiRT_PromptBlock:
    """
    Bundles Positive and Negative Prompts.
    Updates Global Wireless State for Prompts.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("UME_PROMPTS",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/Generation"

    def process(self, positive, negative):
        # 1. Update Global State
        UME_SHARED_STATE[KEY_POSITIVE] = positive
        UME_SHARED_STATE[KEY_NEGATIVE] = negative

        # 2. Return Bundle
        prompts = {
            "positive": positive,
            "negative": negative,
        }
        return (prompts,)

class UmeAiRT_WirelessImageProcess:
    """
    Central node for Wireless Image Editing (Inpaint, Outpaint, Img2Img, Txt2Img).
    Handles Resizing, Padding (Outpaint), Mask Blurring (Inpaint), and Denoise.
    Updates Wireless Global State.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img"}),
            },
            "optional": {
                "image": ("IMAGE",), # Optional input overrides wireless source
                "mask": ("MASK",),   # Optional input overrides wireless mask
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "mask_blur": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                # Outpaint Params
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Wireless/Pre-Process"

    def process_image(self, denoise=1.0, mode="img2img", image=None, mask=None, resize=False, mask_blur=0, 
                      padding_left=0, padding_top=0, padding_right=0, padding_bottom=0, feathering=40):
        
        # Priority logic for txt2img: Ignore input image/mask? 
        # Actually txt2img usually implies we want to GENERATE from scratch using dimensions.
        # But if image is provided, maybe we use it for size?
        # User request: "forcer le denoise a 1 et ignore les option de mask"
        
        
        if mode == "txt2img":
             log_node("Wireless Process: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             # Force Denoise
             UME_SHARED_STATE[KEY_DENOISE] = 1.0
             # Hide Mask
             UME_SHARED_STATE[KEY_SOURCE_MASK] = None
             
             # Image? If provided, maybe user wants to use it as reference sizing or latent upscale?
             # Standard txt2img doesn't use source image in KSampler unless going for img2img loop.
             # If we pass image, Sampler might try to encode it.
             # Let's honor the image update if provided (e.g. for workflow coherence) but Denoise 1.0 makes it effectively ignored by standard KSampler (destruction).
             if image is not None:
                  UME_SHARED_STATE[KEY_SOURCE_IMAGE] = image
             
             return (image, None)

        # 1. Fetch Input (Priority: Wired > Wireless)
        if image is None:
            image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        if mask is None:
            mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)
        
        if image is None:
            UME_SHARED_STATE[KEY_DENOISE] = denoise
            return (None, None)

        B, H, W, C = image.shape
        
        # Determine Target Size (Wireless State)
        target_w = W
        target_h = H
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))

        # Helper: Resize
        def resize_tensor(tensor, tH, tW, interp_mode="bilinear", is_mask=False):
             if is_mask:
                 t = tensor.unsqueeze(1)
             else:
                 t = tensor.permute(0, 3, 1, 2)
             
             t_resized = torch.nn.functional.interpolate(t, size=(tH, tW), mode=interp_mode, align_corners=False if interp_mode!="nearest" else None)
             
             if is_mask:
                 return t_resized.squeeze(1)
             else:
                 return t_resized.permute(0, 2, 3, 1)

        # 2. Process based on Mode
        final_image = image
        final_mask = mask
        
        # RESIZE (Pre-Outpaint)
        if resize:
             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear", is_mask=False)
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)
             
             # Update dims
             B, H, W, C = final_image.shape

        # OUTPAINT
        if mode == "outpaint":
             pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
             
             if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                 # Pad Image (Constant 0)
                 img_p = final_image.permute(0, 3, 1, 2)
                 img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                 final_image = img_padded.permute(0, 2, 3, 1)
                 
                 new_h = H + pad_t + pad_b
                 new_w = W + pad_l + pad_r
                 
                 # Create Mask logic for Outpaint
                 
                 # Start with Zeros (Keep everything)
                 new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)
                 
                 # If original mask existed, pad it
                 if final_mask is not None:
                     # final_mask is [B, H, W]
                     # Check dims
                     if len(final_mask.shape) == 2: m_in = final_mask.unsqueeze(0)
                     else: m_in = final_mask
                     
                     # Pad original mask with 0 (preserve existing mask content)
                     m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                     if len(final_mask.shape) == 2: new_mask = m_padded.squeeze(0)
                     else: new_mask = m_padded

                 # Set Padded Areas to 1.0 (Inpaint)
                 if pad_t > 0: new_mask[:, :pad_t, :] = 1.0
                 if pad_b > 0: new_mask[:, -pad_b:, :] = 1.0
                 if pad_l > 0: new_mask[:, :, :pad_l] = 1.0
                 if pad_r > 0: new_mask[:, :, -pad_r:] = 1.0
                 
                 # Feathering logic
                 if feathering > 0:
                      import torchvision.transforms.functional as TF
                      k = feathering
                      if k % 2 == 0: k += 1
                      sig = float(k) / 3.0
                      
                      if len(new_mask.shape) == 2: m_b = new_mask.unsqueeze(0).unsqueeze(0)
                      else: m_b = new_mask.unsqueeze(1)
                      
                      m_b = TF.gaussian_blur(m_b, kernel_size=k, sigma=sig)
                      
                      if len(new_mask.shape) == 2: new_mask = m_b.squeeze(0).squeeze(0)
                      else: new_mask = m_b.squeeze(1)
                 
                 final_mask = new_mask

        # INPAINT / COMPOSITE BLUR logic preparation
        # Blur mask if requested (Inpaint typically, or cleanup)
        # Note: If Outpaint feathered heavily, manual blur might be redundant but safe.
        
        if mask_blur > 0 and final_mask is not None:
             if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
             elif len(final_mask.shape) == 3: m = final_mask.unsqueeze(1)
             else: m = final_mask
             
             import torchvision.transforms.functional as TF
             k = mask_blur
             if k % 2 == 0: k += 1
             m = TF.gaussian_blur(m, kernel_size=k)
             
             if len(final_mask.shape) == 2: final_mask = m.squeeze(0).squeeze(0)
             elif len(final_mask.shape) == 3: final_mask = m.squeeze(1)

        # MODE HANDLING (State Update)
        state_mask = final_mask
        if mode == "img2img":
            state_mask = None # Hide mask from state
            log_node("Wireless Process: Img2Img Mode (State Mask Hidden).", color="YELLOW")
        
        
        # 6. Update State
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = final_image
        UME_SHARED_STATE[KEY_SOURCE_MASK] = state_mask
        UME_SHARED_STATE[KEY_DENOISE] = denoise

        return (final_image, final_mask)


class UmeAiRT_WirelessImageLoader(comfy_nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_wireless_image"
    CATEGORY = "UmeAiRT/Wireless/Loaders"

    def load_wireless_image(self, image):
        # Load Image (from parent LoadImage)
        out = super().load_image(image)
        img = out[0]
        mask = out[1]

        # Update Wireless Global State (Raw)
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        UME_SHARED_STATE[KEY_SOURCE_MASK] = mask
        
        return (img, mask)


# --- TOOLS NODES ---

class UmeAiRT_Bundle_Downloader:
    """
    Downloads curated model bundles from Hugging Face based on UmeAiRT Auto-Installer scripts.
    """
    @classmethod
    def INPUT_TYPES(s):
        # Load bundles from JSON
        s.bundles_data = {}
        json_path = os.path.join(os.path.dirname(__file__), "umeairt_bundles.json")
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    s.bundles_data = json.load(f)
            except Exception as e:
                print(f"UmeAiRT: Error loading bundles.json: {e}")
        
        categories = list(s.bundles_data.keys()) if s.bundles_data else ["Error: No Bundles"]
        # Collect all unique versions for the dropdown
        all_versions = set()
        for cat in s.bundles_data.values():
            all_versions.update(cat.keys())
        versions = sorted(list(all_versions)) if all_versions else ["None"]
        # Always add Auto
        versions.insert(0, "Auto")

        return {
            "required": {
                "bundle_category": (categories,),
                "bundle_version": (versions,),
            },
            "optional": {
                "hf_token": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_report",)
    FUNCTION = "download_bundle"
    CATEGORY = "UmeAiRT/Tools"
    OUTPUT_NODE = True

    def get_vram_gb(self):
        try:
            import torch
            if torch.cuda.is_available():
                # Returns bytes, convert to GB
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return 0
        except:
            return 0

    def get_best_version(self, category, vram):
        """
        Generic data-driven VRAM selection.
        Iterates all versions in the category, checks 'min_vram' from JSON.
        Returns the version with the highest min_vram that fits within the user's VRAM.
        """
        category_data = self.bundles_data.get(category, {})
        if not category_data:
            return None
            
        compatible_versions = []
        for version, data in category_data.items():
            # Support both old list format (backward compatibility) and new dict format
            if isinstance(data, list):
                # Old format: assume 0 VRAM req or specific hardcoded handling if needed.
                # For now, treat as extremely low req (0 GB).
                min_req = 0 
            else:
                min_req = data.get("min_vram", 0)
            
            if vram >= min_req:
                compatible_versions.append((version, min_req))
        
        if not compatible_versions:
            # Fallback: if nothing fits (unlikely if we have low-end options), 
            # return the one with the lowest requirement.
            all_versions = []
            for version, data in category_data.items():
                 req = 0 if isinstance(data, list) else data.get("min_vram", 0)
                 all_versions.append((version, req))
            if all_versions:
                # Sort by req ascending, return first
                all_versions.sort(key=lambda x: x[1])
                return all_versions[0][0]
            return None

        # Sort by min_vram descending (Greedy approach: use max available resources)
        # If multiple have same min_vram, secondary sort by name (optional)
        compatible_versions.sort(key=lambda x: x[1], reverse=True)
        return compatible_versions[0][0]

    def download_bundle(self, bundle_category, bundle_version, hf_token=""):
        import requests
        import subprocess
        import shutil
        import json # Ensure json is imported if not at top
        from tqdm import tqdm

        # Reload data
        json_path = os.path.join(os.path.dirname(__file__), "umeairt_bundles.json")
        bundles_data = {}
        # Update self.bundles_data locally for this run as well to ensure freshness
        if os.path.exists(json_path):
             with open(json_path, 'r', encoding='utf-8') as f:
                 bundles_data = json.load(f)
                 self.bundles_data = bundles_data

        # --- Base URL ---
        BASE_URL = "https://huggingface.co/UmeAiRT/ComfyUI-Auto_installer/resolve/main/models"
        
        # --- Resolve Paths ---
        try:
            path_checkpoints = folder_paths.get_folder_paths("checkpoints")[0]
            path_clip = folder_paths.get_folder_paths("clip")[0]
            path_vae = folder_paths.get_folder_paths("vae")[0]
            path_upscale = folder_paths.get_folder_paths("upscale_models")[0]
            
            if "controlnet" in folder_paths.folder_names_and_paths:
                path_controlnet = folder_paths.get_folder_paths("controlnet")[0]
            else:
                 path_controlnet = os.path.join(path_checkpoints, "../controlnet")
            
            # Custom subfolders
            # Use parent of checkpoints to get to "models" root
            models_dir = os.path.dirname(path_checkpoints)
            path_flux_diff = os.path.join(models_dir, "diffusion_models", "FLUX")
            path_zimg_diff = os.path.join(models_dir, "diffusion_models", "Z-IMG")
            path_zimg_unet = os.path.join(models_dir, "unet", "Z-IMG")
            
            # Additional Paths for Expanded Bundles
            path_flux_unet = os.path.join(models_dir, "unet", "FLUX")
            path_pulid = os.path.join(models_dir, "pulid")
            path_style = os.path.join(models_dir, "style_models")
            path_lora_flux = os.path.join(models_dir, "loras", "FLUX")
            path_xlabs = os.path.join(models_dir, "xlabs", "controlnets")

            # Path Mapping for JSON
            path_map = {
                "flux_diff": path_flux_diff,
                "flux_unet": path_flux_unet,
                "clip": path_clip,
                "vae": path_vae,
                "zimg_diff": path_zimg_diff,
                "zimg_unet": path_zimg_unet,
                "upscale": path_upscale,
                "controlnet": path_controlnet,
                "xlabs": path_xlabs,
                "pulid": path_pulid,
                "style": path_style,
                "lora_flux": path_lora_flux
            }
            
            
        except Exception as e:
            return (f"Error resolving paths: {e}",)



        # --- Execution Logic ---
        category_data = bundles_data.get(bundle_category, {})
        
        # Handle Auto
        final_version = bundle_version
        if bundle_version == "Auto":
            vram = self.get_vram_gb()
            # print(f"UmeAiRT: Detected {vram:.2f} GB VRAM. Selecting best version...")
            log_node(f"Detected {vram:.2f} GB VRAM. Selecting best version...", color="CYAN")
            best = self.get_best_version(bundle_category, vram)

            if best and best in category_data:
                final_version = best
                # print(f"UmeAiRT: Auto-selected '{final_version}' for '{bundle_category}'")
                log_node(f"Auto-selected '{final_version}' for '{bundle_category}'", color="GREEN")

            else:
                return (f"Auto-detection failed or not supported for {bundle_category}",)



        version_data = category_data.get(final_version)
        
        if not version_data:
             if bundle_category in bundles_data:
                 return (f"Version '{final_version}' not found in '{bundle_category}'",)
             return ("Select a valid bundle to download.",)



        # Handle new object structure vs old list structure
        if isinstance(version_data, list):
            target_list = version_data
        else:
            target_list = version_data.get("files", [])

        if not target_list:
            return (f"No files defined for {bundle_category} - {final_version}",)


        # Initialize Log Buffer
        log_buffer = []
        def log(msg):
            # print(msg)
            # Simple heuristic for color
            color = None
            if "Success" in msg: color = "GREEN"
            elif "Failed" in msg or "Error" in msg: color = "RED"
            elif "Downloading" in msg: color = "CYAN"
            
            log_node(msg, color=color)
            log_buffer.append(msg)



        log(f"UmeAiRT Downloader: Processing {bundle_category} - {final_version}")

        download_count = 0
        skip_count = 0
        
        headers = {}
        if hf_token: headers["Authorization"] = f"Bearer {hf_token}"
        
        # Detect Aria2 (Always try)
        aria2_path = None
        
        # 1. Custom Path (Windows) - Common in UmeAiRT env
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            potential_path = os.path.join(local_app_data, "aria2", "aria2c.exe")
            if os.path.exists(potential_path):
                aria2_path = potential_path
        
        # 2. System Path
        if not aria2_path:
            aria2_path = shutil.which("aria2c")

        if aria2_path:
            # print(f"UmeAiRT: Aria2 detected at {aria2_path}")
            log_node(f"Aria2 detected at {aria2_path}", color="GREEN")
        else:
             # print("UmeAiRT: Aria2 not found. Falling back to requests.")
             log_node("Aria2 not found. Falling back to requests.", color="RED")


        # print(f"UmeAiRT Downloader: Processing {bundle_category} - {final_version}")
        log_node(f"Downloader Processing: {bundle_category} - {final_version}", color="CYAN")


        for item in target_list:
            # Parse JSON Item
            url_suffix = item.get("url")
            path_type = item.get("path_type")
            fname = item.get("filename")
            
            if not url_suffix or not path_type or not fname:
                log(f"Invalid config for {fname}")
                continue

            target_dir = path_map.get(path_type)
            if not target_dir:
                 log(f"Unknown path type: {path_type}")
                 continue

            if not os.path.exists(target_dir):
                try: os.makedirs(target_dir, exist_ok=True)
                except OSError as e:
                     log(f"Mkdir Fail: {e}")
                     continue

            full_path = os.path.join(target_dir, fname)
            
            # Check if file exists, but also check for typical aria2 incomplete file
            if os.path.exists(full_path):
                if os.path.exists(full_path + ".aria2"):
                    log(f"Resuming incomplete download (aria2): {fname}")
                else:
                    log(f"Skipping (Exists): {fname}")
                    skip_count += 1
                    continue
            
            # Download
            if url_suffix.startswith("http"):
                full_url = url_suffix
            else:
                full_url = BASE_URL + url_suffix
                
            log(f"Downloading: {fname} ...")
            
            success = False
            
            # --- Try Aria2 ---
            if aria2_path:
                try:
                    # -x 16 -s 16 -k 1M -d "dir" -o "filename"
                    # --allow-overwrite=true if needed, but we check existence before
                    cmd = [
                        aria2_path,
                        "-c",
                        "-x", "16", "-s", "16", "-k", "1M",
                        "--console-log-level=error",
                        "--summary-interval=0",
                        "-d", target_dir,
                        "-o", fname,
                        full_url
                    ]
                    # Add auth token if present (syntax: --header="Authorization: Bearer ...")
                    if hf_token:
                         cmd.append(f"--header=Authorization: Bearer {hf_token}")
                    
                    subprocess.run(cmd, check=True)
                    log(f"  [Aria2] Success: {fname}")
                    success = True
                except subprocess.CalledProcessError as e:
                    log(f"  [Aria2] Failed: {e}. Trying request fallback...")
                except Exception as e:
                    log(f"  [Aria2] Error: {e}. Trying request fallback...")

            # --- Fallback to Requests ---
            if not success:
                try:
                    with requests.get(full_url, stream=True, headers=headers) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        with open(full_path, 'wb') as f, tqdm(
                            desc=fname,
                            total=total_size,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for chunk in r.iter_content(chunk_size=8192):
                                size = f.write(chunk)
                                bar.update(size)
                    success = True
                    log(f"  [Requests] Success: {fname}")
                except Exception as e:
                    log(f"  [Requests] Failed: {e}")
                    if os.path.exists(full_path): os.remove(full_path)
            
            if success:
                download_count += 1
            else:
                log(f"FATAL: Could not download {fname}")

        summary = f"Done. Downloaded: {download_count}, Skipped: {skip_count}, Total Checked: {len(target_list)}"
        log(summary)
        
        full_log = "\n".join(log_buffer)
        return (full_log,)



class UmeAiRT_Log_Viewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "log_text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_output",)
    FUNCTION = "show_log"
    CATEGORY = "UmeAiRT/Tools"
    OUTPUT_NODE = True

    def show_log(self, log_text):
        return {"ui": {"text": [log_text]}, "result": (log_text,)}


class UmeAiRT_BlockImageProcess:
    """
    Central node for Block Image Editing.
    Outputs UME_IMAGE bundle for Block Sampler.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img"}),
            },
            "optional": {
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "mask_blur": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                # Outpaint Params
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Blocks/Images"

    def process_image(self, image_bundle, denoise=0.75, mode="img2img", resize=False, mask_blur=0, 
                      padding_left=0, padding_top=0, padding_right=0, padding_bottom=0, feathering=40):
        
        # Unpack Bundle
        image = image_bundle.get("image")
        mask = image_bundle.get("mask") # Can be None

        if image is None:
            raise ValueError("UmeAiRT: Block Image Process received a bundle with no image.")

        # TXT2IMG Logic
        if mode == "txt2img":
             print("UmeAiRT Block Process: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).")
             denoise = 1.0
             mask = None 
             # Just pass through image logic (potentially for size reference if resized) but mask is killed.
        

        # 1. Base Logic
        B, H, W, C = image.shape
        
        # Determine Target Size
        target_w = W
        target_h = H
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))

        # 2. Process
        final_image = image
        final_mask = mask
        
        # RESIZE
        if resize:
             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear", is_mask=False)
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)
             
             B, H, W, C = final_image.shape

        # OUTPAINT
        if mode == "outpaint":
             pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
             
             if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                 # Pad Image (Constant 0)
                 img_p = final_image.permute(0, 3, 1, 2)
                 img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                 final_image = img_padded.permute(0, 2, 3, 1)
                 
                 new_h = H + pad_t + pad_b
                 new_w = W + pad_l + pad_r
                 
                 new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)
                 
                 # Pad original mask
                 if final_mask is not None:
                     if len(final_mask.shape) == 2: m_in = final_mask.unsqueeze(0)
                     else: m_in = final_mask
                     
                     m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                     if len(final_mask.shape) == 2: new_mask = m_padded.squeeze(0)
                     else: new_mask = m_padded

                 # Set Padded Areas to 1.0
                 if pad_t > 0: new_mask[:, :pad_t, :] = 1.0
                 if pad_b > 0: new_mask[:, -pad_b:, :] = 1.0
                 if pad_l > 0: new_mask[:, :, :pad_l] = 1.0
                 if pad_r > 0: new_mask[:, :, -pad_r:] = 1.0
                 
                 # Feathering
                 if feathering > 0:
                      import torchvision.transforms.functional as TF
                      k = feathering
                      if k % 2 == 0: k += 1
                      sig = float(k) / 3.0
                      
                      if len(new_mask.shape) == 2: m_b = new_mask.unsqueeze(0).unsqueeze(0)
                      else: m_b = new_mask.unsqueeze(1)
                      m_b = TF.gaussian_blur(m_b, kernel_size=k, sigma=sig)
                      if len(new_mask.shape) == 2: new_mask = m_b.squeeze(0).squeeze(0)
                      else: new_mask = m_b.squeeze(1)
                 
                 final_mask = new_mask

        # INPAINT / BLUR
        if (mode == "inpaint" or mode == "outpaint") and final_mask is not None:
             if mask_blur > 0:
                 if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
                 elif len(final_mask.shape) == 3: m = final_mask.unsqueeze(1)
                 else: m = final_mask
                 
                 import torchvision.transforms.functional as TF
                 k = mask_blur
                 if k % 2 == 0: k += 1
                 m = TF.gaussian_blur(m, kernel_size=k)
                 
                 if len(final_mask.shape) == 2: final_mask = m.squeeze(0).squeeze(0)
                 elif len(final_mask.shape) == 3: final_mask = m.squeeze(1)
        
        # Mode Handling for Sampler
        final_mode = "img2img"
        state_mask = final_mask
        
        if mode == "txt2img":
             final_mode = "txt2img"
             final_mask = None
             state_mask = None
        elif mode == "img2img":
             final_mask = None # For Bundle
             state_mask = None # For Wireless State
             print("UmeAiRT Block Process: Img2Img Mode (Masks Hidden).")
        elif mode == "outpaint":
             final_mode = "inpaint" # Sampler treats outpaint as inpaint
        elif mode == "inpaint":
             final_mode = "inpaint"

        # Create Bundle
        image_bundle = {
            "image": final_image,
            "mask": final_mask,
            "mode": final_mode,
            "denoise": denoise,
        }

        # Also update Wireless state for synergy
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = final_image
        UME_SHARED_STATE[KEY_SOURCE_MASK] = state_mask
        UME_SHARED_STATE[KEY_DENOISE] = denoise

        return (image_bundle,)


class UmeAiRT_BlockImageLoader(comfy_nodes.LoadImage):
    """
    Block version of Image Loader.
    Outputs UME_IMAGE bundle (Default mode=img2img, denoise=0.75).
    For raw outputs + bundle, use 'Block Image Loader (Advanced)'.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "load_block_image"
    CATEGORY = "UmeAiRT/Blocks/Images"

    def load_block_image(self, image):
        # Load Image (from parent LoadImage)
        out = super().load_image(image)
        img = out[0]
        mask = out[1]
        
        # Also update Wireless state for synergy (Raw)
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        UME_SHARED_STATE[KEY_SOURCE_MASK] = mask

        # Create Default Bundle
        image_bundle = {
            "image": img,
            "mask": mask,
            "mode": "img2img",
            "denoise": 0.75, # Standard Default
        }

        # Return Bundle ONLY
        return (image_bundle,)


class UmeAiRT_BlockImageLoader_Advanced(comfy_nodes.LoadImage):
    """
    Advanced Block Image Loader.
    Outputs UME_IMAGE bundle + Raw Image + Raw Mask.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("UME_IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image_bundle", "image", "mask")
    FUNCTION = "load_block_image"
    CATEGORY = "UmeAiRT/Blocks/Images"

    def load_block_image(self, image):
        # Load Image (from parent LoadImage)
        out = super().load_image(image)
        img = out[0]
        mask = out[1]
        
        # Also update Wireless state for synergy (Raw)
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        UME_SHARED_STATE[KEY_SOURCE_MASK] = mask

        # Create Default Bundle
        image_bundle = {
            "image": img,
            "mask": mask,
            "mode": "img2img",
            "denoise": 0.75,
        }

        return (image_bundle, img, mask)


class UmeAiRT_BlockSampler:
    """
    Sampler that takes 'Block' bundles as input.
    Prioritizes inputs from the 'settings', 'files', and 'prompts' bundles.
    Falls back to 'Wireless' state if something is missing in the bundle (or bundle not linked).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "models": ("UME_FILES",),
                "prompts": ("UME_PROMPTS",),
                "settings": ("UME_SETTINGS",),
                "loras": ("UME_LORA_STACK",),
                "image": ("UME_IMAGE",),  # Changed from IMAGE to UME_IMAGE bundle
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/Samplers"

    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()
        self.cnet_loader = comfy_nodes.ControlNetLoader()
        self.cnet_apply = comfy_nodes.ControlNetApplyAdvanced()

    def process(self, settings, models=None, loras=None, prompts=None, image=None):
        # 0. Extract ControlNets from Image Bundle (Linear Workflow)
        controlnets = []
        if image and isinstance(image, dict):
            controlnets = image.get("controlnets", [])

        # 1. Expand Files/Models (Prioritize Bundle > Wireless)
        if models:
            model = models.get("model")
            vae = models.get("vae")
            clip = models.get("clip")
        else:
            model = UME_SHARED_STATE.get(KEY_MODEL)
            vae = UME_SHARED_STATE.get(KEY_VAE)
            clip = UME_SHARED_STATE.get(KEY_CLIP)

        # 1.5 Apply LoRA Stack (if present)
        if loras:
            # We need valid model/clip first
            if not model or not clip:
                raise ValueError("UmeAiRT Block Sampler: Cannot apply LoRAs without a base Model/CLIP (from 'models' or Wireless state).")
            
            # Reset LoRA list metadata for this run
            loaded_loras_meta = [] 

            for lora_def in loras:
                # Format: (lora_name, strength_model, strength_clip)
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                         model, clip = self.lora_loader.load_lora(model, clip, name, str_model, str_clip)
                         loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        print(f"UmeAiRT Sampler Warning: Failed to apply LoRA {name}: {e}")

            # Update Wireless State with PATCHED model/clip and Metadata
            UME_SHARED_STATE[KEY_MODEL] = model
            UME_SHARED_STATE[KEY_CLIP] = clip
            UME_SHARED_STATE[KEY_LORAS] = loaded_loras_meta
        
        # Ensure we have everything
        if not model or not vae or not clip:
            missing = []
            if not model: missing.append("MODEL")
            if not vae: missing.append("VAE")
            if not clip: missing.append("CLIP")
            raise ValueError(f"❌ Block Sampler: Missing {', '.join(missing)}!\n\n💡 Solution: Connect a 'Model Loader (Block)' node to the 'models' input, or add a 'Wireless Checkpoint Loader' node.")

        # 2. Expand Settings (Prioritize Bundle > Wireless)
        if settings:
            width = settings.get("width", 1024)
            height = settings.get("height", 1024)
            steps = settings.get("steps", 20)
            cfg = settings.get("cfg", 8.0)
            sampler_name = settings.get("sampler", "euler")
            scheduler = settings.get("scheduler", "normal")
            seed = settings.get("seed", 0)
        else:
            # Fallback to Wireless state
            size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
            width = size.get("width", 1024)
            height = size.get("height", 1024)
            steps = UME_SHARED_STATE.get(KEY_STEPS, 20)
            cfg = UME_SHARED_STATE.get(KEY_CFG, 8.0)
            sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
            scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
            seed = UME_SHARED_STATE.get(KEY_SEED, 0)
        
        # Denoise: Get from image bundle if present, otherwise 1.0 (Txt2Img)
        if image is not None and isinstance(image, dict):
            denoise = image.get("denoise", 1.0)
        else:
            denoise = 1.0

        # 3. Expand Prompts (Prioritize Bundle > Wireless)
        if prompts:
            pos_text = prompts.get("positive", "")
            neg_text = prompts.get("negative", "")
        else:
            pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
            neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        # 4. Latent Logic (Image Input -> VAE Encode OR Empty Latent)
        latent_image = None
        
        # Priority 1: Wired UME_IMAGE Bundle (Img2Img / Inpaint)
        if image is not None:
            try:
                # Extract from bundle
                if isinstance(image, dict):
                    # UME_IMAGE bundle format
                    raw_image = image.get("image")
                    source_mask = image.get("mask")
                    mode = image.get("mode", "img2img")
                else:
                    # Legacy: raw IMAGE tensor
                    raw_image = image
                    source_mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)
                    mode = "img2img"
                
                # VAE Encode
                latent_image = comfy_nodes.VAEEncode().encode(vae, raw_image)[0]
                
                 # Apply mask for Inpaint mode
                if source_mask is not None:
                    if torch.any(source_mask > 0):
                        latent_image["noise_mask"] = source_mask
            except Exception as e:
                print(f"UmeAiRT Block Error: VAE Encode failed: {e}")

        # Priority 2: Wireless Latent (Fallback if no image wired)
        if latent_image is None:
            wireless_latent = UME_SHARED_STATE.get(KEY_LATENT)
            if wireless_latent is not None:
                latent_image = wireless_latent

        # Priority 3: Empty Latent (Txt2Img)
        if latent_image is None:
            batch_size = 1
            l = torch.zeros([batch_size, 4, height // 8, width // 8])
            latent_image = {"samples": l}

        # 5. Encode Prompts
        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]

        # 5.5 Apply ControlNets (if present)
        # We need to apply them to the conditioning (positive/negative)
        if controlnets:
            for cnet_def in controlnets:
                # Format: (cnet_name, image, strength, start, end)
                c_name, c_image, c_str, c_start, c_end = cnet_def
                
                if c_name != "None" and c_image is not None:
                    try:
                        # Load ControlNet Model
                        cnet_model = self.cnet_loader.load_controlnet(c_name)[0]
                        
                        # Apply to Positive & Negative (Advanced)
                        positive, negative = self.cnet_apply.apply_controlnet(
                            positive, negative, cnet_model, c_image, c_str, c_start, c_end
                        )
                        
                    except Exception as e:
                        print(f"UmeAiRT Sampler Warning: Failed to apply ControlNet {c_name}: {e}")

        # 6. Sample
        mode_str = "img2img" if image is not None else "txt2img"
        if image is not None and isinstance(image, dict) and image.get("mode") == "inpaint":
            mode_str = "inpaint"
        
        print(f"\n{'='*60}")
        print(f"🎨 UmeAiRT Block Sampler")
        print(f"{'='*60}")
        print(f"  Mode: {mode_str}")
        print(f"  Size: {width}x{height}")
        print(f"  Steps: {steps} | CFG: {cfg} | Denoise: {denoise}")
        print(f"  Sampler: {sampler_name} | Scheduler: {scheduler}")
        print(f"  Seed: {seed}")
        print(f"  Positive: {pos_text}")
        print(f"  Negative: {neg_text}")
        print(f"{'='*60}\n")
        
        result_latent = comfy_nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]

        # 7. Decode
        generated_image = comfy_nodes.VAEDecode().decode(vae, result_latent)[0]

        # 8. Auto-Composite for Inpaint
        # If we are in inpaint mode (mask present), we almost always want to composite the result
        # onto the original image to ensure seamless edges, especially since we have the source image and mask.
        
        if mode_str == "inpaint" and image is not None:
             # We need raw_source_image and source_mask
             # They are retrieved earlier: raw_image, source_mask
             
             if raw_image is not None and source_mask is not None:
                 try:
                     # Resize Source Image & Mask to match Generated Image [B, H, W, C]
                     B, H, W, C = generated_image.shape
                     
                     # --- Source Resize ---
                     source_resized = raw_image
                     sB, sH, sW, sC = source_resized.shape
                     if sH != H or sW != W:
                         s_p = source_resized.permute(0, 3, 1, 2)
                         s_resized = torch.nn.functional.interpolate(s_p, size=(H, W), mode="bilinear", align_corners=False)
                         source_resized = s_resized.permute(0, 2, 3, 1)

                     # --- Mask Resize ---
                     mask_resized = source_mask
                     if len(mask_resized.shape) == 2:
                         mask_resized = mask_resized.unsqueeze(0)
                     
                     mB, mH, mW = mask_resized.shape
                     if mH != H or mW != W:
                         m_p = mask_resized.unsqueeze(1)
                         m_resized = torch.nn.functional.interpolate(m_p, size=(H, W), mode="nearest") # Mask uses nearest usually, but Composite node used bilinear? 
                         # Let's use bilinear for softer mask edges during resize if it was already blurred?
                         # Actually standard resize for mask is usually nearest to preserve crispness, 
                         # BUT here we want soft edges. Let's stick to bilinear for mask resize too to be safe/smooth.
                         m_resized = torch.nn.functional.interpolate(m_p, size=(H, W), mode="bilinear", align_corners=False)
                         mask_resized = m_resized.squeeze(1)

                     # --- Composite ---
                     m = mask_resized
                     if len(m.shape) == 3:
                         m = m.unsqueeze(-1)
                     elif len(m.shape) == 2:
                         m = m.unsqueeze(0).unsqueeze(-1)
                     
                     if m.shape[0] < B:
                         m = m.repeat(B, 1, 1, 1)
                     
                     if source_resized.shape[0] < B:
                         source_resized = source_resized.repeat(B, 1, 1, 1)

                     m = torch.clamp(m, 0.0, 1.0)
                     
                     # Result = Source * (1-M) + Generated * M
                     composite_image = source_resized * (1.0 - m) + generated_image * m
                     
                     print("🎨 UmeAiRT Block Inpaint: Automatically composited result with source.")
                     return (composite_image,)

                 except Exception as e:
                     print(f"UmeAiRT Block Warning: Auto-Composite failed ({e}). Returning generated image.")
                     return (generated_image,)
        
        return (generated_image,)


class UmeAiRT_BlockUltimateSDUpscale(UmeAiRT_WirelessUltimateUpscale_Base):
    """
    Block version of UltimateSDUpscale.
    Accepts settings/models/loras/prompts bundles with fallback to Wireless.
    """
    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
            },
            "optional": {
                "settings": ("UME_SETTINGS",),
                "models": ("UME_FILES",),
                "loras": ("UME_LORA_STACK",),
                "prompts": ("UME_PROMPTS",),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Global Prompt"}),
                "mode_type": (usdu_modes, {"default": "Linear"}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Blocks/Post-Process"

    def upscale(self, image, model, upscale_by, settings=None, models=None, loras=None, prompts=None, 
                denoise=0.35, clean_prompt=True, mode_type="Linear", tile_padding=32):
        # Note: 'model' here is the upscale model filename, 'models' is the SD models bundle
        upscale_model_file = model  # Rename to avoid confusion with SD model
        
        # 1. Expand Models (Prioritize Bundle > Wireless)
        if models:
            sd_model = models.get("model")
            vae = models.get("vae")
            clip = models.get("clip")
        else:
            sd_model = UME_SHARED_STATE.get(KEY_MODEL)
            vae = UME_SHARED_STATE.get(KEY_VAE)
            clip = UME_SHARED_STATE.get(KEY_CLIP)

        # 2. Apply LoRA Stack
        if loras:
            if not sd_model or not clip:
                raise ValueError("❌ Block Upscale: Cannot apply LoRAs without Model/CLIP!\n\n💡 Solution: Connect a 'Model Loader (Block)' node to the 'models' input first.")
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                        sd_model, clip = self.lora_loader.load_lora(sd_model, clip, name, str_model, str_clip)
                        loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        print(f"Block USDU Warning: LoRA {name} failed: {e}")
            UME_SHARED_STATE[KEY_MODEL] = sd_model
            UME_SHARED_STATE[KEY_CLIP] = clip
            UME_SHARED_STATE[KEY_LORAS] = loaded_loras_meta

        if not sd_model or not vae or not clip:
            missing = []
            if not sd_model: missing.append("MODEL")
            if not vae: missing.append("VAE")
            if not clip: missing.append("CLIP")
            raise ValueError(f"❌ Block Upscale: Missing {', '.join(missing)}!\n\n💡 Solution: Connect a 'Model Loader (Block)' node to the 'models' input, or add a 'Wireless Checkpoint Loader' node.")

        # 3. Expand Settings
        if settings:
            steps = settings.get("steps", 20)
            cfg = settings.get("cfg", 1.0)
            sampler_name = settings.get("sampler", "euler")
            scheduler = settings.get("scheduler", "normal")
            seed = settings.get("seed", 0)
            tile_width = settings.get("width", 512)
            tile_height = settings.get("height", 512)
        else:
            steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
            cfg = 1.0  # Default for USDU
            sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
            scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
            seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
            size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 512, "height": 512})
            tile_width = int(size.get("width", 512))
            tile_height = int(size.get("height", 512))

        # 4. Expand Prompts
        if prompts:
            pos_text = prompts.get("positive", "")
            neg_text = prompts.get("negative", "")
        else:
            pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
            neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        # Clean Prompt Logic
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)

        # 5. Load Upscale Model
        try:
            from comfy_extras.nodes_upscale_model import UpscaleModelLoader
            upscale_model = UpscaleModelLoader().load_model(upscale_model_file)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        # 6. Execute USDU
        usdu_node = self.get_usdu_node()
        # denoise is passed as arg
        steps = max(5, steps // 4)  # 1/4 of normal steps
        mask_blur = 16

        return usdu_node.upscale(
            image=image, model=sd_model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=tile_width, tile_height=tile_height, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode="None", seam_fix_denoise=1.0,
            seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
            force_uniform_tiles=True, tiled_decode=False,
            suppress_preview=True,
        )


class UmeAiRT_BlockFaceDetailer(UmeAiRT_WirelessUltimateUpscale_Base):
    """
    Block version of FaceDetailer.
    Accepts settings/models/loras/prompts bundles with fallback to Wireless.
    """
    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (folder_paths.get_filename_list("bbox"),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                "settings": ("UME_SETTINGS",),
                "models": ("UME_FILES",),
                "loras": ("UME_LORA_STACK",),
                "prompts": ("UME_PROMPTS",),
                "guide_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "max_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Blocks/Post-Process"

    def face_detail(self, image, model, denoise, settings=None, models=None, loras=None, prompts=None, guide_size=512, max_size=1024):
        # Note: 'model' here is the bbox detector filename, 'models' is the SD models bundle
        bbox_model_file = model  # Rename to avoid confusion with SD model
        
        # 1. Expand Models
        if models:
            sd_model = models.get("model")
            vae = models.get("vae")
            clip = models.get("clip")
        else:
            sd_model = UME_SHARED_STATE.get(KEY_MODEL)
            vae = UME_SHARED_STATE.get(KEY_VAE)
            clip = UME_SHARED_STATE.get(KEY_CLIP)

        # 2. Apply LoRA Stack
        if loras:
            if not sd_model or not clip:
                raise ValueError("❌ Block FaceDetailer: Cannot apply LoRAs without Model/CLIP!\n\n💡 Solution: Connect a 'Model Loader (Block)' node to the 'models' input first.")
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                        sd_model, clip = self.lora_loader.load_lora(sd_model, clip, name, str_model, str_clip)
                        loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        print(f"Block FaceDetailer Warning: LoRA {name} failed: {e}")
            UME_SHARED_STATE[KEY_MODEL] = sd_model
            UME_SHARED_STATE[KEY_CLIP] = clip
            UME_SHARED_STATE[KEY_LORAS] = loaded_loras_meta

        if not sd_model or not vae or not clip:
            missing = []
            if not sd_model: missing.append("MODEL")
            if not vae: missing.append("VAE")
            if not clip: missing.append("CLIP")
            raise ValueError(f"❌ Block FaceDetailer: Missing {', '.join(missing)}!\n\n💡 Solution: Connect a 'Model Loader (Block)' node to the 'models' input, or add a 'Wireless Checkpoint Loader' node.")

        # 3. Expand Settings
        if settings:
            steps = settings.get("steps", 20)
            cfg = settings.get("cfg", 8.0)
            sampler_name = settings.get("sampler", "euler")
            scheduler = settings.get("scheduler", "normal")
            seed = settings.get("seed", 0)
        else:
            steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
            cfg = float(UME_SHARED_STATE.get(KEY_CFG, 8.0))
            sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
            scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
            seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))

        # 4. Expand Prompts
        if prompts:
            pos_text = prompts.get("positive", "")
            neg_text = prompts.get("negative", "")
        else:
            pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
            neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        positive, negative = self.encode_prompts(clip, pos_text, neg_text)

        # 5. Import & Load Detector
        try:
            from .facedetailer_core import logic as fd_logic
            from .facedetailer_core import detector
        except ImportError:
            import facedetailer_core.logic as fd_logic
            import facedetailer_core.detector as detector

        bbox_detector = detector.load_bbox_model(bbox_model_file)

        # 6. Detect & Detail
        bbox_threshold = 0.5
        bbox_dilation = 10
        bbox_crop_factor = 3.0
        drop_size = 10
        feather = 5
        noise_mask = True
        force_inpaint = True
        guide_size_for_bbox = True

        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

        result = fd_logic.do_detail(
            image=image, segs=segs, model=sd_model, clip=clip, vae=vae,
            guide_size=guide_size, guide_size_for_bbox=guide_size_for_bbox, max_size=max_size,
            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative, denoise=denoise,
            feather=feather, noise_mask=noise_mask, force_inpaint=force_inpaint,
            drop_size=drop_size
        )

        return result

# --- UNPACK NODES (Bundle -> Raw) ---

class UmeAiRT_Unpack_ImageBundle:
    """
    Unpacks UME_IMAGE bundle into Image and Mask.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
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
        
        # If mask is None, return empty mask? Or just None?
        # ComfyUI usually handles None gracefully in some nodes, but let's be safe.
        # If mask is missing, standard nodes might expect it.
        # For now, return what is in the bundle.
        
        return (image, mask)

class UmeAiRT_Unpack_FilesBundle:
    """
    Unpacks UME_FILES bundle into Model, Clip, VAE, and Name.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models_bundle": ("UME_FILES",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, models_bundle):
        if not isinstance(models_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_FILES bundle.")
        
        return (
            models_bundle.get("model"),
            models_bundle.get("clip"),
            models_bundle.get("vae"),
            models_bundle.get("model_name", "")
        )

class UmeAiRT_Unpack_SettingsBundle:
    """
    Unpacks UME_SETTINGS bundle into individual parameters.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "settings_bundle": ("UME_SETTINGS",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "steps", "cfg", "denoise", "seed", "sampler_name", "scheduler", "guidance")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, settings_bundle):
        if not isinstance(settings_bundle, dict):
             raise ValueError("UmeAiRT Unpack: Input is not a valid UME_SETTINGS bundle.")

        return (
            settings_bundle.get("width", 1024),
            settings_bundle.get("height", 1024),
            settings_bundle.get("steps", 20),
            settings_bundle.get("cfg", 8.0),
            settings_bundle.get("denoise", 1.0), # Often not in settings bundle, default to 1.0
            settings_bundle.get("seed", 0),
            settings_bundle.get("sampler", "euler"),
            settings_bundle.get("scheduler", "normal"),
            settings_bundle.get("cfg", 8.0), # guidance alias
        )


class UmeAiRT_Faces_Unpack_Node:
    """
    Unpacks FACES_DATA bundle? Or just pass through?
    Wait, assuming Input is UME_FACES?
    If not defined elsewhere, assuming generic structure.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "faces_bundle": ("UME_FACES",),
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
    Unpacks UME_TAGS bundle?
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tags_bundle": ("UME_TAGS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_string",)
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, tags_bundle):
        # Assuming tags_bundle is either dict or list or string
        # Default behavior: stringify
        return (str(tags_bundle),)

class UmeAiRT_Pipe_Unpack_Node:
    """
    Unpacks UME_PIPE bundle (Model, Clip, Vae, Positive, Negative, Latent, etc.)
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe_bundle": ("UME_PIPE",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "positive", "negative")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, pipe_bundle):
         if not isinstance(pipe_bundle, dict):
             # Try fallback tuple if from ImpactPack
             if isinstance(pipe_bundle, (list, tuple)) and len(pipe_bundle) >= 5:
                  return (pipe_bundle[0], pipe_bundle[1], pipe_bundle[2], pipe_bundle[3], pipe_bundle[4])
             raise ValueError("UmeAiRT Unpack: Input is not a valid UME_PIPE bundle.")
         
         return (
             pipe_bundle.get("model"),
             pipe_bundle.get("clip"),
             pipe_bundle.get("vae"),
             # Extract strings from conditioning? Or raw text?
             # Standard Basic Pipe usually carries raw text in custom implementations
             pipe_bundle.get("positive_text", ""),
             pipe_bundle.get("negative_text", "")
         )

class UmeAiRT_Unpack_PromptsBundle:
    """
    Unpacks UME_PROMPTS bundle into Positive and Negative strings.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompts_bundle": ("UME_PROMPTS",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "unpack"
    CATEGORY = "UmeAiRT/Unpack"

    def unpack(self, prompts_bundle):
        if not isinstance(prompts_bundle, dict):
            raise ValueError("UmeAiRT Unpack: Input is not a valid UME_PROMPTS bundle.")

        return (
            prompts_bundle.get("positive", ""),
            prompts_bundle.get("negative", "")
        )

class UmeAiRT_Seed_Node:
    """
    Standard Seed Node with UI control.
    Sets the Wireless SEED value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "set_seed"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_seed(self, seed):
        UME_SHARED_STATE[KEY_SEED] = seed
        return (seed,)

class UmeAiRT_CR_Seed_Node:
    """
    Alternative Seed Node (ComfyRoll style or similar).
    Also sets the Wireless SEED value.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "set_seed"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_seed(self, seed):
        UME_SHARED_STATE[KEY_SEED] = seed
        return (seed,)

# Mapping
NODE_CLASS_MAPPINGS = {
    "UmeAiRT_Guidance_Input": UmeAiRT_Guidance_Input,
    "UmeAiRT_Guidance_Output": UmeAiRT_Guidance_Output,
    "UmeAiRT_ImageSize_Input": UmeAiRT_ImageSize_Input,
    "UmeAiRT_ImageSize_Output": UmeAiRT_ImageSize_Output,
    "UmeAiRT_FPS_Input": UmeAiRT_FPS_Input,
    "UmeAiRT_FPS_Output": UmeAiRT_FPS_Output,
    "UmeAiRT_Steps_Input": UmeAiRT_Steps_Input,
    "UmeAiRT_Steps_Output": UmeAiRT_Steps_Output,
    "UmeAiRT_Seed_Input": UmeAiRT_Seed_Input,
    "UmeAiRT_Seed_Output": UmeAiRT_Seed_Output,
    "UmeAiRT_Denoise_Input": UmeAiRT_Denoise_Input,
    "UmeAiRT_Denoise_Output": UmeAiRT_Denoise_Output,
    "UmeAiRT_Sampler_Input": UmeAiRT_Sampler_Input,
    "UmeAiRT_Sampler_Output": UmeAiRT_Sampler_Output,
    "UmeAiRT_Scheduler_Input": UmeAiRT_Scheduler_Input,
    "UmeAiRT_Scheduler_Output": UmeAiRT_Scheduler_Output,
    "UmeAiRT_Seed_Node": UmeAiRT_Seed_Node,
    "UmeAiRT_CR_Seed_Node": UmeAiRT_CR_Seed_Node,
    "UmeAiRT_WirelessKSampler": UmeAiRT_WirelessKSampler,
    "UmeAiRT_WirelessUltimateUpscale": UmeAiRT_WirelessUltimateUpscale,
    "UmeAiRT_WirelessUltimateUpscale_Advanced": UmeAiRT_WirelessUltimateUpscale_Advanced,
    "UmeAiRT_Unified_ControlNet": UmeAiRT_ControlNetImageProcess,
    "UmeAiRT_Wireless_Image_Loader": UmeAiRT_WirelessImageLoader,
    "UmeAiRT_BlockImageProcess": UmeAiRT_BlockImageProcess,
    "UmeAiRT_BlockSampler": UmeAiRT_BlockSampler,
    "UmeAiRT_Faces_Unpack_Node": UmeAiRT_Faces_Unpack_Node,
    "UmeAiRT_Tags_Unpack_Node": UmeAiRT_Tags_Unpack_Node,
    "UmeAiRT_Pipe_Unpack_Node": UmeAiRT_Pipe_Unpack_Node,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,
    "UmeAiRT_Unpack_PromptsBundle": UmeAiRT_Unpack_PromptsBundle, 
    "UmeAiRT_Bundle_Downloader": UmeAiRT_Bundle_Downloader,
    "UmeAiRT_Log_Viewer": UmeAiRT_Log_Viewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmeAiRT_Guidance_Input": "UmeAiRT Guide In",
    "UmeAiRT_Guidance_Output": "UmeAiRT Guide Out",
    "UmeAiRT_ImageSize_Input": "UmeAiRT Size In",
    "UmeAiRT_ImageSize_Output": "UmeAiRT Size Out",
    "UmeAiRT_FPS_Input": "UmeAiRT FPS In",
    "UmeAiRT_FPS_Output": "UmeAiRT FPS Out",
    "UmeAiRT_Steps_Input": "UmeAiRT Step In",
    "UmeAiRT_Steps_Output": "UmeAiRT Step Out",
    "UmeAiRT_Seed_Input": "UmeAiRT Seed In",
    "UmeAiRT_Seed_Output": "UmeAiRT Seed Out",
    "UmeAiRT_Denoise_Input": "UmeAiRT Denoise In",
    "UmeAiRT_Denoise_Output": "UmeAiRT Denoise Out",
    "UmeAiRT_Sampler_Input": "UmeAiRT Sampler In",
    "UmeAiRT_Sampler_Output": "UmeAiRT Sampler Out",
    "UmeAiRT_Scheduler_Input": "UmeAiRT Scheduler In",
    "UmeAiRT_Scheduler_Output": "UmeAiRT Scheduler Out",
    "UmeAiRT_Seed_Node": "🌱 UmeAiRT Seed Node",
    "UmeAiRT_CR_Seed_Node": "🌱 UmeAiRT CR Seed Node",
    "UmeAiRT_WirelessKSampler": "📡 UmeAiRT Wireless KSampler",
    "UmeAiRT_WirelessUltimateUpscale": "📡 UmeAiRT Wireless Ultimate Upscale",
    "UmeAiRT_WirelessUltimateUpscale_Advanced": "📡 UmeAiRT Wireless Ultimate Upscale (Adv)",
    "UmeAiRT_Unified_ControlNet": "🕹️ UmeAiRT Unified ControlNet",
    "UmeAiRT_Wireless_Image_Loader": "📡 UmeAiRT Wireless Image Loader",
    "UmeAiRT_BlockImageProcess": "🧱 UmeAiRT Block Image Process",
    "UmeAiRT_BlockSampler": "🧱 UmeAiRT Block Sampler",
    "UmeAiRT_Faces_Unpack_Node": "📦 UmeAiRT Faces Unpack",
    "UmeAiRT_Tags_Unpack_Node": "📦 UmeAiRT Tags Unpack",
    "UmeAiRT_Pipe_Unpack_Node": "📦 UmeAiRT Pipe Unpack",
    "UmeAiRT_Unpack_SettingsBundle": "📦 UmeAiRT Settings Unpack",
    "UmeAiRT_Unpack_PromptsBundle": "📦 UmeAiRT Prompts Unpack",
    "UmeAiRT_Bundle_Downloader": "⬇️ UmeAiRT Bundle Downloader",
    "UmeAiRT_Log_Viewer": "📜 UmeAiRT Log Viewer"
}
