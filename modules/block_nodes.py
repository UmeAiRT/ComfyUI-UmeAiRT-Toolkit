import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import (
    UME_SHARED_STATE, KEY_MODEL, KEY_CLIP, KEY_VAE, KEY_LORAS, KEY_CONTROLNETS,
    KEY_POSITIVE, KEY_NEGATIVE, KEY_LATENT, KEY_SEED, KEY_STEPS, KEY_CFG,
    KEY_SAMPLER, KEY_SCHEDULER, KEY_DENOISE, KEY_SOURCE_IMAGE, KEY_SOURCE_MASK,
    KEY_IMAGESIZE, KEY_MODEL_NAME, resize_tensor, log_node
)
from .logger import logger
from .logic_nodes import UmeAiRT_WirelessUltimateUpscale_Base
from .optimization_utils import SamplerContext

try:
    from .facedetailer_core import detector, logic as fd_logic
except ImportError:
    pass

# --- Helper for LoRA Stacks ---

def get_lora_inputs(count):
    inputs = {
        "required": {},
        "optional": {
            "loras": ("UME_LORA_STACK", {"tooltip": "Optional input to chain multiple LoRA stacks."}),
        }
    }
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    for i in range(1, count + 1):
        inputs["optional"][f"lora_{i}_name"] = (lora_list, {"tooltip": f"Select LoRA model {i}."})
        inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": f"Strength for LoRA {i}."})
    return inputs

def process_lora_stack(loras, **kwargs):
    current_stack = []
    if loras:
        current_stack.extend(loras)
    
    indices = set()
    for k in kwargs.keys():
        if k.startswith("lora_") and "_name" in k:
            parts = k.split("_")
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
    def INPUT_TYPES(s): return get_lora_inputs(1)
    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"
    def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_3:
    @classmethod
    def INPUT_TYPES(s): return get_lora_inputs(3)
    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"
    def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_5:
    @classmethod
    def INPUT_TYPES(s): return get_lora_inputs(5)
    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"
    def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)

class UmeAiRT_LoraBlock_10:
    @classmethod
    def INPUT_TYPES(s): return get_lora_inputs(10)
    RETURN_TYPES = ("UME_LORA_STACK",)
    RETURN_NAMES = ("loras",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/LoRA"
    def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)


# --- ControlNet Blocks ---

class UmeAiRT_ControlNetImageApply_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input Image Bundle."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"tooltip": "Select ControlNet model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "optional_control_image": ("IMAGE", {"tooltip": "Optional: Override control image."}), 
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
            
            # (name, image, strength, start, end)
            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent))
            
        new_bundle["controlnets"] = cnet_stack
        UME_SHARED_STATE[KEY_CONTROLNETS] = cnet_stack

        return (new_bundle,)

class UmeAiRT_ControlNetImageApply_Simple(UmeAiRT_ControlNetImageApply_Advanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider"}),
            }
        }
    def apply_controlnet(self, image_bundle, control_net_name, strength):
        return super().apply_controlnet(image_bundle, control_net_name, strength, 0.0, 1.0, None)

class UmeAiRT_ControlNetImageProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["img2img", "txt2img"], {"default": "img2img"}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
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
        if not isinstance(image_bundle, dict): raise ValueError("ControlNet Image Process: Input is not a valid UME_IMAGE bundle.")
        
        image = image_bundle.get("image")
        mask = image_bundle.get("mask")
        
        if image is None: raise ValueError("ControlNet Image Process: Bundle has no image.")

        if mode == "txt2img":
             log_node("Unified CNet: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             denoise = 1.0
             mask = None

        final_image = image
        final_mask = mask
        
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))
             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear")
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)

        final_mode = "img2img"
        if mode == "txt2img":
             final_mode = "txt2img"
             final_mask = None
        elif mode == "img2img":
             final_mask = None

        new_bundle = {
            "image": final_image,
            "mask": final_mask,
            "mode": final_mode,
            "denoise": denoise,
            "controlnets": image_bundle.get("controlnets", []).copy() if image_bundle.get("controlnets") else []
        }
        
        cnet_stack = new_bundle["controlnets"]
        if control_net_name != "None":
            cnet_stack.append((control_net_name, final_image, strength, 0.0, 1.0))
        
        new_bundle["controlnets"] = cnet_stack
        UME_SHARED_STATE[KEY_CONTROLNETS] = cnet_stack

        return (new_bundle,)


# --- Parameter Blocks ---


class UmeAiRT_GenerationSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "tooltip": "Target width of the generated image."}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64, "display": "slider", "tooltip": "Target height of the generated image."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1, "display": "slider", "tooltip": "Total sampling steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5, "display": "slider", "tooltip": "CFG Scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise scheduler."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Seed for random number generation."}),
            }
        }
    RETURN_TYPES = ("UME_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Blocks/Generation"

    def process(self, width, height, steps, cfg, sampler_name, scheduler, seed):
        UME_SHARED_STATE[KEY_IMAGESIZE] = {"width": width, "height": height}
        UME_SHARED_STATE[KEY_STEPS] = steps
        UME_SHARED_STATE[KEY_CFG] = cfg
        UME_SHARED_STATE[KEY_SAMPLER] = sampler_name
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        UME_SHARED_STATE[KEY_SEED] = seed
        return ({
            "width": width, "height": height, "steps": steps, "cfg": cfg,
            "sampler": sampler_name, "scheduler": scheduler, "seed": seed
        },)


# --- Files / Model Loaders (Block) ---

class UmeAiRT_FilesSettings_Checkpoint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }
    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "load"
    CATEGORY = "UmeAiRT/Blocks/Loaders"
    
    def load(self, ckpt_name):
        model, clip, vae = comfy_nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = ckpt_name
        
        log_node(f"Block Checkpoint Loaded: {ckpt_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": ckpt_name},)

class UmeAiRT_FilesSettings_FLUX(UmeAiRT_FilesSettings_Checkpoint):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
            }
        }
    def load(self, unet_name, clip_name1, clip_name2, vae_name):
         # Helper to load models manually
         # Using standard comfy nodes
         from nodes import UNETLoader, CLIPLoader, VAELoader, DualCLIPLoader
         
         model = UNETLoader().load_unet(unet_name, "default")[0]
         
         # Dual Clip Logic
         # If clip1 is t5 and clip2 is clip_l, use DualCLIPLoader? Or direct?
         # Comfy's DualCLIPLoader expects (clip_name1, clip_name2, type)
         
         try:
             clip = DualCLIPLoader().load_clip(clip_name1, clip_name2, "flux")[0]
         except:
             # Fallback if specific node fails or user selects wrong types
             # Just load clip1?
             clip = CLIPLoader().load_clip(clip_name1)[0]
             
         vae = VAELoader().load_vae(vae_name)[0]
         
         UME_SHARED_STATE[KEY_MODEL] = model
         UME_SHARED_STATE[KEY_CLIP] = clip
         UME_SHARED_STATE[KEY_VAE] = vae
         UME_SHARED_STATE[KEY_MODEL_NAME] = unet_name
         
         log_node(f"Block FLUX Loaded: {unet_name}", color="GREEN")
         return ({"model": model, "clip": clip, "vae": vae, "model_name": unet_name},)

class UmeAiRT_FilesSettings_Fragmented:
    @classmethod
    def INPUT_TYPES(s):
        # Combined list for flexible loading
        models = folder_paths.get_filename_list("checkpoints") + folder_paths.get_filename_list("diffusion_models")
        clips = folder_paths.get_filename_list("checkpoints") + folder_paths.get_filename_list("text_encoders") # Sometimes clip in ckpt?
        vaes = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "model_name": (models,),
                "clip_name": (clips,),
                "vae_name": (vaes,),
            }
        }
    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "load"
    CATEGORY = "UmeAiRT/Blocks/Loaders"
    
    def load(self, model_name, clip_name, vae_name):
        from nodes import CheckpointLoaderSimple, UNETLoader, CLIPLoader, VAELoader
        
        # Smart Load Model
        # Try Checkpoint first, then UNET
        model = None
        clip = None
        vae = None
        
        # 1. Model
        if model_name in folder_paths.get_filename_list("checkpoints"):
             m, c, v = CheckpointLoaderSimple().load_checkpoint(model_name)
             model = m
             # If clip/vae are "None" or same, use these?
             # But user selected explicit clip/vae. Use those.
        else:
             model = UNETLoader().load_unet(model_name, "default")[0]
             
        # 2. Clip
        if clip_name != "None":
             clip = CLIPLoader().load_clip(clip_name)[0]
        elif clip is None:
             # If we loaded checkpoint, we have clip.
             # If we loaded UNET, we need clip.
             pass

        # 3. VAE
        if vae_name != "None":
             vae = VAELoader().load_vae(vae_name)[0]
             
        # Fallback for Checkpoint Loading
        if model is None: raise ValueError("Fragmented Loader: Failed to load model.")
        if clip is None: raise ValueError("Fragmented Loader: Failed to load CLIP.")
        if vae is None: raise ValueError("Fragmented Loader: Failed to load VAE.")

        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = model_name
        
        log_node(f"Block Fragmented Loaded: {model_name}", color="GREEN")
        return ({"model": model, "clip": clip, "vae": vae, "model_name": model_name},)


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
                "unet_name": (folder_paths.get_filename_list("unet"), {"tooltip": "Select the UNET model file."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"], {"tooltip": "Weight data type for loading the UNET (affects VRAM usage)."}),
                "clip_name1": (folder_paths.get_filename_list("clip"), {"tooltip": "Select the primary CLIP model (e.g., t5xxl)."}),
                "clip_name2": (folder_paths.get_filename_list("clip"), {"tooltip": "Select the secondary CLIP model (e.g., clip_l)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "Select the VAE model file."}),
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
        except:
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

    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks/Models"

    def load_files(self, model_name, clip_name, vae_name, weight_dtype="default", clip_type="stable_diffusion", clip_skip=-1, device="default"):
        # 1. Load Model (Checkpoint/UNET)
        # Determine path and type
        ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
        diff_path = folder_paths.get_full_path("diffusion_models", model_name)
        unet_path = folder_paths.get_full_path("unet", model_name)

        model = None
        
        # Priority: Diffusion Models/UNET -> Checkpoints
        if diff_path or unet_path:
            # It's a Diffusion Model / UNET
            final_path = diff_path if diff_path else unet_path
            
            # Use load_diffusion_model with dtype options
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            
            # Using comfy.sd.load_diffusion_model directly matches native UNETLoader
            model = comfy.sd.load_diffusion_model(final_path, model_options=model_options)
            log_node(f"Fragmented Model (UNET) Loaded: {model_name} [{weight_dtype}]", color="GREEN")

        elif ckpt_path:
            # It's a Checkpoint
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model = out[0]
            log_node(f"Fragmented Model (Checkpoint) Loaded: {model_name}", color="GREEN")
        else:
            raise ValueError(f"Fragmented Loader: Model '{model_name}' not found.")
        
        # 2. Load CLIP (Text Encoder)
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if clip_path is None:
            try:
                clip_path = folder_paths.get_full_path("text_encoders", clip_name)
            except:
                pass
        
        if clip_path is None:
            raise ValueError(f"Fragmented Loader: Could not find CLIP file '{clip_name}' in 'clip' or 'text_encoders' folders.")

        # Prepare CLIP Type & Options
        clip_type_enum = getattr(comfy.sd.CLIPType, clip_type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        clip_options = {}
        if device == "cpu":
            clip_options["load_device"] = clip_options["offload_device"] = torch.device("cpu")

        # Load CLIP with type and options
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type_enum, model_options=clip_options)

        # 3. Load VAE
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 4. CLIP Skip
        if clip_skip != -1:
             clip = clip.clone()
             clip.clip_layer(clip_skip)

        # 5. Update Global State
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = model_name
        UME_SHARED_STATE[KEY_LORAS] = []

        # 6. Return Bundle
        files = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "model_name": model_name
        }
        log_node(f"Block Fragmented Loaded: {model_name}", color="GREEN")
        return (files,)

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
        except Exception:
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

    RETURN_TYPES = ("UME_FILES",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_files"
    CATEGORY = "UmeAiRT/Blocks/Loaders"

    def load_files(self, model_name, clip_name, vae_name):
        # 1. Load Model (Z-IMG / Diffusion Model)
        if model_name.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import UnetLoaderGGUF
            model = UnetLoaderGGUF().load_unet(model_name)[0]
            log_node(f"Z-IMG Model Loader: Loaded '{model_name}' [Type: GGUF]")
        else:
            diff_path = folder_paths.get_full_path("diffusion_models", model_name)
            if not diff_path:
                raise ValueError(f"Z-IMG Loader: Model '{model_name}' not found in 'diffusion_models'.")

            # Auto-Detect Weight DType
            model_options = {}
            detected_dtype = "default"
            
            if "e4m3fn" in model_name.lower():
                model_options["dtype"] = torch.float8_e4m3fn
                detected_dtype = "fp8_e4m3fn"
            elif "e5m2" in model_name.lower():
                model_options["dtype"] = torch.float8_e5m2
                detected_dtype = "fp8_e5m2"
            
            # Load Model
            model = comfy.sd.load_diffusion_model(diff_path, model_options=model_options)
            # Verify actual Dtype using Safetensors metadata natively if not Forced
            detected_dtype = "Unquantized"
            log_node(f"Z-IMG Model Loader: Loaded '{model_name}' [Auto-DType: {detected_dtype}]")
        
        # 2. Load CLIP (Hardcoded to LUMINA2 or loaded via GGUF)
        if clip_name.endswith(".gguf"):
            from ..vendor.comfyui_gguf.gguf_nodes import CLIPLoaderGGUF
            clip = CLIPLoaderGGUF().load_clip(clip_name, type="lumina2")[0]
            log_node(f"Z-IMG CLIP Loader: Loaded '{clip_name}' [Type: GGUF LUMINA2]")
        else:
            clip_path = folder_paths.get_full_path("clip", clip_name)
            if clip_path is None:
                try:
                    clip_path = folder_paths.get_full_path("text_encoders", clip_name)
                except:
                     pass
            
            if clip_path is None:
                raise ValueError(f"Z-IMG Loader: Could not find CLIP file '{clip_name}'.")

            # Hardcoded LUMINA2 type
            clip_type_enum = comfy.sd.CLIPType.LUMINA2
            # Default device options
            clip_options = {}

            # Load CLIP
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type_enum, model_options=clip_options)
            log_node(f"Z-IMG CLIP Loader: Loaded '{clip_name}' [Type: LUMINA2]")

        # 3. Load VAE
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 4. Update Global State
        UME_SHARED_STATE[KEY_MODEL] = model
        UME_SHARED_STATE[KEY_CLIP] = clip
        UME_SHARED_STATE[KEY_VAE] = vae
        UME_SHARED_STATE[KEY_MODEL_NAME] = model_name
        UME_SHARED_STATE[KEY_LORAS] = []

        # 5. Return Bundle
        files = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "model_name": model_name
        }
        return (files,)


# --- Image Blocks ---

class UmeAiRT_BlockImageLoader(comfy_nodes.LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "load_block_image"
    CATEGORY = "UmeAiRT/Blocks/Images"

    def load_block_image(self, image):
        out = super().load_image(image)
        img, mask = out[0], out[1]
        
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        UME_SHARED_STATE[KEY_SOURCE_MASK] = mask

        image_bundle = {"image": img, "mask": mask, "mode": "img2img", "denoise": 0.75}
        return (image_bundle,)

class UmeAiRT_BlockImageLoader_Advanced(UmeAiRT_BlockImageLoader):
    RETURN_TYPES = ("UME_IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image_bundle", "image", "mask")
    def load_block_image(self, image):
        res = super().load_block_image(image)
        return (res[0], res[0]["image"], res[0]["mask"])

class UmeAiRT_BlockImageProcess:
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
                "mask_blur": ("INT", {"default": 10}),
                "padding_left": ("INT", {"default": 0}), "padding_top": ("INT", {"default": 0}),
                "padding_right": ("INT", {"default": 0}), "padding_bottom": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Blocks/Images"

    def process_image(self, image_bundle, denoise=0.75, mode="img2img", resize=False, mask_blur=0, 
                      padding_left=0, padding_top=0, padding_right=0, padding_bottom=0):
        
        image = image_bundle.get("image")
        mask = image_bundle.get("mask")
        
        if image is None: raise ValueError("Block Image Process: Bundle has no image.")

        if mode == "txt2img":
             log_node("Block Process: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             denoise = 1.0
             mask = None 

        B, H, W, C = image.shape
        target_w, target_h = W, H
        
        final_image, final_mask = image, mask
        
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))

             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear")
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)
             
             B, H, W, C = final_image.shape

        if mode == "outpaint":
             pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
             if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                 img_p = final_image.permute(0, 3, 1, 2)
                 # Use 'replicate' to stretch edge pixels outward instead of 'constant' black bars
                 img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
                 final_image = img_padded.permute(0, 2, 3, 1)
                 
                 new_h = H + pad_t + pad_b
                 new_w = W + pad_l + pad_r
                 new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)
                 
                 if final_mask is not None:
                     if len(final_mask.shape) == 2: m_in = final_mask.unsqueeze(0)
                     else: m_in = final_mask
                     m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                     if len(final_mask.shape) == 2: new_mask = m_padded.squeeze(0)
                     else: new_mask = m_padded

                 # Set Padded Areas to 1.0 (Inpaint) - ComfyUI standard: 1.0 = generate, 0.0 = keep
                 # We add a slight overlap (8 px) into the original image so the AI can blend the edge seamlessly
                 overlap = 8
                 if pad_t > 0: new_mask[:, :pad_t + overlap, :] = 1.0
                 if pad_b > 0: new_mask[:, -(pad_b + overlap):, :] = 1.0
                 if pad_l > 0: new_mask[:, :, :pad_l + overlap] = 1.0
                 if pad_r > 0: new_mask[:, :, -(pad_r + overlap):] = 1.0
                 
                 feathering = 40
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
                 
                 
                 # The 'replicate' padding above provides the edge colors for the outpaint region.
                 # The mask ensures those areas are re-generated. No need for VAEEncodeForInpaint here, 
                 # as it destructively turns the padded areas into 50% grey!

        if (mode == "inpaint" or mode == "outpaint") and final_mask is not None and mask_blur > 0:
             import torchvision.transforms.functional as TF
             if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
             else: m = final_mask
             k = mask_blur
             if k % 2 == 0: k += 1
             m = TF.gaussian_blur(m, kernel_size=k)
             final_mask = m.squeeze(0).squeeze(0) if len(final_mask.shape) == 2 else m

        final_mode = "inpaint" if mode in ["inpaint", "outpaint"] else "img2img"
        if mode == "txt2img": final_mode = "txt2img"
        elif mode == "img2img": final_mask = None

        state_mask = final_mask if mode != "img2img" else None

        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = final_image
        UME_SHARED_STATE[KEY_SOURCE_MASK] = state_mask
        UME_SHARED_STATE[KEY_DENOISE] = denoise

        return ({"image": final_image, "mask": final_mask, "mode": final_mode, "denoise": denoise},)


# --- Processor Blocks ---

class UmeAiRT_BlockSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("UME_FILES",), 
                "settings": ("UME_SETTINGS",),
                "positive": ("POSITIVE", {"forceInput": True}),
            },
            "optional": {
                "negative": ("NEGATIVE", {"forceInput": True}),
                "loras": ("UME_LORA_STACK",), 
                "image": ("UME_IMAGE",), 
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
        
        # Local Prompt Cache
        self._last_pos_text = None
        self._last_neg_text = None
        self._last_clip = None
        self._cached_positive = None
        self._cached_negative = None

    def process(self, settings=None, models=None, loras=None, positive=None, negative=None, image=None):
        controlnets = []
        if image and isinstance(image, dict):
            controlnets = image.get("controlnets", [])

        if models:
            model, vae, clip = models.get("model"), models.get("vae"), models.get("clip")
        else:
            model, vae, clip = UME_SHARED_STATE.get(KEY_MODEL), UME_SHARED_STATE.get(KEY_VAE), UME_SHARED_STATE.get(KEY_CLIP)

        if loras:
            if not model or not clip: raise ValueError("Block Sampler: No base Model/CLIP for LoRAs.")
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                         model, clip = self.lora_loader.load_lora(model, clip, name, str_model, str_clip)
                         loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        log_node(f"Block Sampler LoRA Error ({name}): {e}", color="RED")
            
            UME_SHARED_STATE[KEY_MODEL] = model
            UME_SHARED_STATE[KEY_CLIP] = clip
            UME_SHARED_STATE[KEY_LORAS] = loaded_loras_meta
        
        if not model or not vae or not clip: raise ValueError("Block Sampler: Missing Model/VAE/CLIP.")

        if settings:
            width, height = settings.get("width", 1024), settings.get("height", 1024)
            steps, cfg = settings.get("steps", 20), settings.get("cfg", 8.0)
            sampler_name, scheduler = settings.get("sampler", "euler"), settings.get("scheduler", "normal")
            seed = settings.get("seed", 0)
        else:
            size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
            width, height = int(size.get("width", 1024)), int(size.get("height", 1024))
            steps = UME_SHARED_STATE.get(KEY_STEPS, 20)
            cfg = UME_SHARED_STATE.get(KEY_CFG, 8.0)
            sampler_name, scheduler = UME_SHARED_STATE.get(KEY_SAMPLER, "euler"), UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
            seed = UME_SHARED_STATE.get(KEY_SEED, 0)
        
        denoise = image.get("denoise", 1.0) if image else 1.0

        # Handle Prompts (Discrete Inputs fallback to Wireless)
        pos_text = positive if positive is not None else str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = negative if negative is not None else str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        latent_image = None
        mode_str = "txt2img"
        raw_image, source_mask = None, None
        
        if image:
             raw_image = image.get("image")
             source_mask = image.get("mask")
             mode_str = image.get("mode", "img2img")
             
             if mode_str in ["inpaint", "outpaint"] and source_mask is not None:
                 # For inpaint/outpaint, we MUST encode the image and apply the noise_mask
                 # even if denoise is 1.0, otherwise the whole image gets regenerated.
                 latent_image = comfy_nodes.VAEEncode().encode(vae, raw_image)[0]
                 latent_image["noise_mask"] = source_mask
             elif denoise < 1.0:
                 latent_image = comfy_nodes.VAEEncode().encode(vae, raw_image)[0]

        if latent_image is None:
             wireless_latent = UME_SHARED_STATE.get(KEY_LATENT)
             if wireless_latent is not None: latent_image = wireless_latent

        if latent_image is None:
             l = torch.zeros([1, 4, height // 8, width // 8], device="cpu")
             latent_image = {"samples": l}
             denoise = 1.0

        if self._last_pos_text == pos_text and self._last_neg_text == neg_text and self._last_clip is clip:
             positive_cond = self._cached_positive
             negative_cond = self._cached_negative
             log_node("Block Sampler: Using cached Prompts (Fast Start)", color="GREEN")
        else:
             log_node("Block Sampler: Encoding Prompts...")
             tokens = clip.tokenize(pos_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             positive_cond = [[cond, {"pooled_output": pooled}]]

             tokens = clip.tokenize(neg_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             negative_cond = [[cond, {"pooled_output": pooled}]]
             
             self._last_pos_text = pos_text
             self._last_neg_text = neg_text
             self._last_clip = clip
             self._cached_positive = positive_cond
             self._cached_negative = negative_cond

        if controlnets:
            for cnet_def in controlnets:
                c_name, c_image, c_str, c_start, c_end = cnet_def
                if c_name != "None" and c_image is not None:
                    try:
                        c_model = self.cnet_loader.load_controlnet(c_name)[0]
                        positive_cond, negative_cond = self.cnet_apply.apply_controlnet(positive_cond, negative_cond, c_model, c_image, c_str, c_start, c_end)
                    except Exception as e: log_node(f"Block Sampler ControlNet Error: {e}", color="RED")

        log_node(f"Block Sampler: {mode_str} | {width}x{height} | Steps: {steps} | CFG: {cfg}")
        
        from .optimization_utils import warmup_vae
        warmup_vae(vae, latent_image)
        
        with SamplerContext():
             result_latent = comfy_nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive_cond, negative_cond, latent_image, denoise)[0]

        log_node("Block Sampler: Decoding VAE (First run may pause for VRAM loading & compilation)...")
        image_out = comfy_nodes.VAEDecode().decode(vae, result_latent)[0]

        if mode_str == "inpaint" and raw_image is not None and source_mask is not None:
             try:
                 B, H, W, C = image_out.shape
                 source_resized = resize_tensor(raw_image, H, W, interp_mode="bilinear") # Ensure match
                 mask_resized = resize_tensor(source_mask, H, W, interp_mode="bilinear", is_mask=True)
                 
                 m = mask_resized
                 if len(m.shape) == 2: m = m.unsqueeze(0).unsqueeze(-1)
                 elif len(m.shape) == 3: m = m.unsqueeze(-1)
                 
                 if m.shape[0] < B: m = m.repeat(B, 1, 1, 1)
                 if source_resized.shape[0] < B: source_resized = source_resized.repeat(B, 1, 1, 1)
                 
                 m = torch.clamp(m, 0.0, 1.0)
                 image_out = source_resized * (1.0 - m) + image_out * m
                 log_node("Block Inpaint: Auto-Composited.", color="GREEN")
             except Exception as e:
                 log_node(f"Block Inpaint Composite Failed: {e}", color="RED")

        return (image_out,)

class UmeAiRT_BlockUltimateSDUpscale(UmeAiRT_WirelessUltimateUpscale_Base):
    def __init__(self): self.lora_loader = comfy_nodes.LoraLoader()
    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        return {
            "required": { 
                "image": ("IMAGE",), 
                "model": (folder_paths.get_filename_list("upscale_models"),), 
                "upscale_by": ("FLOAT", {"default": 2.0}),
                "settings": ("UME_SETTINGS",), 
                "models": ("UME_FILES",), 
            },
            "optional": { 
                "prompts": ("UME_PROMPTS",),
                "loras": ("UME_LORA_STACK",), 
                "denoise": ("FLOAT", {"default": 0.35}), 
                "clean_prompt": ("BOOLEAN", {"default": True}), 
                "mode_type": (usdu_modes, {"default": "Linear"}), 
                "tile_padding": ("INT", {"default": 32}), 
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Blocks/Post-Processing"

    def upscale(self, image, model, upscale_by, settings=None, models=None, loras=None, prompts=None, denoise=0.35, clean_prompt=True, mode_type="Linear", tile_padding=32):
        if models: sd_model, vae, clip = models.get("model"), models.get("vae"), models.get("clip")
        else: sd_model, vae, clip = UME_SHARED_STATE.get(KEY_MODEL), UME_SHARED_STATE.get(KEY_VAE), UME_SHARED_STATE.get(KEY_CLIP)

        if loras:
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                     try: sd_model, clip = self.lora_loader.load_lora(sd_model, clip, name, str_model, str_clip)
                     except: pass
            UME_SHARED_STATE[KEY_MODEL], UME_SHARED_STATE[KEY_CLIP] = sd_model, clip

        if not sd_model or not vae or not clip: raise ValueError("Block Upscale: Missing Model/VAE/CLIP")

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
             cfg = 1.0
             sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
             scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
             seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
             
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 512, "height": 512})
             if not isinstance(size, dict): size = {"width": 512, "height": 512}
             tile_width = int(size.get("width", 512))
             tile_height = int(size.get("height", 512))

        if prompts: pos_text, neg_text = prompts.get("positive"), prompts.get("negative")
        else: pos_text, neg_text = str(UME_SHARED_STATE.get(KEY_POSITIVE)), str(UME_SHARED_STATE.get(KEY_NEGATIVE))

        positive, negative = self.encode_prompts(clip, "" if clean_prompt else pos_text, neg_text)
        
        try: from comfy_extras.nodes_upscale_model import UpscaleModelLoader; upscale_model = UpscaleModelLoader().load_model(model)[0]
        except: raise ImportError("UpscaleModelLoader not found")

        with SamplerContext():
             res = self.get_usdu_node().upscale(
                 image=image, model=sd_model, positive=positive, negative=negative, vae=vae,
                 upscale_by=upscale_by, seed=seed, steps=max(5, steps//4), cfg=cfg,
                 sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
                 upscale_model=upscale_model, mode_type=mode_type,
                 tile_width=tile_width, tile_height=tile_height, mask_blur=16, tile_padding=tile_padding,
                 seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
                 force_uniform_tiles=True, tiled_decode=False, suppress_preview=True
             )
        return res

class UmeAiRT_BlockFaceDetailer(UmeAiRT_WirelessUltimateUpscale_Base):
    def __init__(self): self.lora_loader = comfy_nodes.LoraLoader()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",), 
                "model": (folder_paths.get_filename_list("bbox"),), 
                "denoise": ("FLOAT", {"default": 0.5}),
                "settings": ("UME_SETTINGS",), 
                "models": ("UME_FILES",), 
            },
            "optional": { 
                "prompts": ("UME_PROMPTS",),
                "loras": ("UME_LORA_STACK",), 
                "guide_size": ("INT", {"default": 512}), 
                "max_size": ("INT", {"default": 1024}), 
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Blocks/Post-Processing"

    def face_detail(self, image, model, denoise, settings=None, models=None, loras=None, prompts=None, guide_size=512, max_size=1024):
        if models: sd_model, vae, clip = models.get("model"), models.get("vae"), models.get("clip")
        else: sd_model, vae, clip = UME_SHARED_STATE.get(KEY_MODEL), UME_SHARED_STATE.get(KEY_VAE), UME_SHARED_STATE.get(KEY_CLIP)

        if loras:
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                     try: sd_model, clip = self.lora_loader.load_lora(sd_model, clip, name, str_model, str_clip)
                     except: pass
            UME_SHARED_STATE[KEY_MODEL], UME_SHARED_STATE[KEY_CLIP] = sd_model, clip

        if not sd_model or not vae or not clip: raise ValueError("Block FaceDetailer: Missing Model/VAE/CLIP")

        if settings: steps, cfg, sampler_name, scheduler, seed = settings.get("steps", 20), settings.get("cfg", 8.0), settings.get("sampler", "euler"), settings.get("scheduler", "normal"), settings.get("seed", 0)
        else: steps, cfg, sampler_name, scheduler, seed = int(UME_SHARED_STATE.get(KEY_STEPS, 20)), float(UME_SHARED_STATE.get(KEY_CFG, 8.0)), UME_SHARED_STATE.get(KEY_SAMPLER, "euler"), UME_SHARED_STATE.get(KEY_SCHEDULER, "normal"), int(UME_SHARED_STATE.get(KEY_SEED, 0))

        if prompts: pos_text, neg_text = prompts.get("positive"), prompts.get("negative")
        else: pos_text, neg_text = str(UME_SHARED_STATE.get(KEY_POSITIVE)), str(UME_SHARED_STATE.get(KEY_NEGATIVE))

        positive, negative = self.encode_prompts(clip, pos_text, neg_text)
        
        try: 
            bbox_detector = detector.load_bbox_model(model)
            segs = bbox_detector.detect(image, 0.5, 10, 3.0, 10)
            
            with SamplerContext():
                return fd_logic.do_detail(
                    image=image, segs=segs, model=sd_model, clip=clip, vae=vae,
                    guide_size=guide_size, guide_size_for_bbox=True, max_size=max_size,
                    seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                    positive=positive, negative=negative, denoise=denoise,
                    feather=5, noise_mask=True, force_inpaint=True, drop_size=10
                )
        except Exception as e:
            log_node(f"FaceDetailer Error: {e}", color="RED")
            return (image,)
