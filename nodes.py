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
import numpy as np
from PIL import Image

import torch
import folder_paths
import nodes as comfy_nodes
import comfy.samplers
import comfy.utils

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Loaders"
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
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_wireless"
    CATEGORY = "UmeAiRT/Loaders"
    OUTPUT_NODE = True

    def load_image_wireless(self, image):
        # Call original loader
        out = super().load_image(image)
        # Set Wireless State
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = out[0]
        UME_SHARED_STATE[KEY_SOURCE_MASK] = out[1]
        return out

class UmeAiRT_SourceImage_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Variables"
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
    CATEGORY = "UmeAiRT/Variables"

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
    CATEGORY = "UmeAiRT/Generation"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Always run if upstream variables changed (which we can't track easily without linking).
        # But since we access globals that fluctuate, we return NaN to force update.
        return float("nan")

    def process(self, signal=None):
        # 1. Fetch Objects
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        if model is None: raise ValueError("Wireless KSampler: No Model linked (Use Model Input)")
        if vae is None: raise ValueError("Wireless KSampler: No VAE linked (Use VAE Input)")
        if clip is None: raise ValueError("Wireless KSampler: No CLIP linked (Use CLIP Input)")

        # 2. Fetch Params
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
        cfg = float(UME_SHARED_STATE.get(KEY_CFG, 8.0))
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        denoise = float(UME_SHARED_STATE.get(KEY_DENOISE, 1.0))
        
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        # 3. Handle Latent (Auto-Detect Txt2Img vs Img2Img)
        wireless_latent = UME_SHARED_STATE.get(KEY_LATENT)
        latent_image = None
        
        # Check for Img2Img via Denoise
        if denoise < 1.0:
            # Try to fetch source image for Img2Img
            source_image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
            if source_image is not None:
                print(f"UmeAiRT DEBUG: Wireless KSampler triggered Img2Img (Denoise {denoise}). Encoding Source Image.")
                # VAE Encode Logic
                # VAEEncode needs (pixels, vae) -> returns (latent,)
                try:
                    latent_image = comfy_nodes.VAEEncode().encode(vae, source_image)[0]
                except Exception as e:
                     print(f"UmeAiRT Error: Failed to VAE Encode source image: {e}")
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

        # Negative
        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]

        # 5. Sample
        # We reuse the standard KSampler function
        return comfy_nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)


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
    CATEGORY = "UmeAiRT/Tools"

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
    CATEGORY = "UmeAiRT/Tools"
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
        
        print(info) # Also print to console
        
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
            raise ValueError("UmeAiRT Wireless USDU: Model, VAE, or CLIP is missing from wireless state.")
            
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
                "upscale_model_name": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Generation"

    def upscale(self, image, enabled, upscale_model_name, upscale_by):
        print(f"DEBUG: UmeAiRT Wireless Upscale Simple - Enabled: {enabled}")
        if not enabled:
            return (image,)

        # Load Upscale Model Internally
        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(upscale_model_name)[0]
        except ImportError:
             raise ImportError("UmeAiRT: Could not import UpscaleModelLoader from comfy_extras. Ensure ComfyUI is standard.")
             
        usdu_node = self.get_usdu_node()
        model, vae, clip, pos_text, neg_text, seed, gen_steps, sampler_name, scheduler, t_w, t_h, wireless_cfg, wireless_denoise = self.fetch_wireless_common()
        positive, negative = self.encode_prompts(clip, pos_text, neg_text)
        
        # Logic: 1/4 steps, rounded up
        steps = math.ceil(gen_steps / 4)
        
        # Hardcoded Simple Settings
        cfg = 1.0
        denoise = 0.35
        mode_type = "Linear"
        mask_blur = 16
        tile_padding = 32
        seam_fix_mode = "None"
        seam_fix_denoise = 1.0

        return usdu_node.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=t_w, tile_height=t_h, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
            force_uniform_tiles=True, tiled_decode=False,
            suppress_preview=True, # Explicitly disable tile previews
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
                "bbox_model_name": (folder_paths.get_filename_list("bbox"),),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "face_detail_simple"
    CATEGORY = "UmeAiRT/Generation"

    def face_detail_simple(self, image, enabled, bbox_model_name, denoise):
        
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
        model, vae, clip, pos_text, neg_text, wireless_seed, wireless_steps, wireless_sampler, wireless_scheduler, _, _, wireless_cfg, _ = self.fetch_wireless_common()
        positive, negative = self.encode_prompts(clip, pos_text, neg_text)

        # Import Logic & Detector
        try:
            from .facedetailer_core import logic as fd_logic
            from .facedetailer_core import detector
        except ImportError:
            import facedetailer_core.logic as fd_logic
            import facedetailer_core.detector as detector

        # Load/Run Detector internally
        bbox_detector = detector.load_bbox_model(bbox_model_name)
        
        # Detect
        # print(f"DEBUG: Running Simple Detector with {bbox_model_name}")
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

        # Run Detailer
        result = fd_logic.do_detail(
            image=image, segs=segs, model=model, clip=clip, vae=vae,
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
    CATEGORY = "UmeAiRT/Tools"

    def save_images(self, images, filename, prompt=None, extra_pnginfo=None):
        
        # Split 'filename' input into path and filename
        # Ensure separators are standardized
        full_pattern = filename.replace("\\", "/")
        if "/" in full_pattern:
             path, filename = full_pattern.rsplit("/", 1)
        else:
             path = ""
             filename = full_pattern
             
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
                print(f"UmeAiRT Error processing LoRAs: {e}")

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
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Generation"

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



