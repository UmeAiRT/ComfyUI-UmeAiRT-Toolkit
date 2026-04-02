import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, UmeSettings, UmeImage, resize_tensor, apply_outpaint_padding, log_node
import torchvision.transforms.functional as TF
from .logger import logger
from typing import Tuple, Dict, Any, Optional, List

def get_lora_inputs(count):
    inputs = {
        "required": {},
        "optional": {
            "loras": ("UME_LORA_STACK", {"tooltip": "Optional input to chain multiple LoRA stacks."}),
        }
    }
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    for i in range(1, count + 1):
        inputs["optional"][f"lora_{i}_on"] = ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off", "tooltip": f"Toggle LoRA {i} on or off."})
        inputs["optional"][f"lora_{i}_name"] = (lora_list, {"tooltip": f"Select LoRA model {i}."})
        inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider", "tooltip": f"Strength for LoRA {i}."})
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
        is_on = kwargs.get(f"lora_{i}_on", True)
        name = kwargs.get(f"lora_{i}_name")
        strength = kwargs.get(f"lora_{i}_strength", 1.0)
        
        if is_on and name and name != "None":
            # Unified strength for model and clip
            current_stack.append((name, strength, strength))
            
    return (current_stack,)

def _make_lora_block_class(count):
    """Factory to create LoRA Block node classes with a given slot count."""
    class _LoraBlock:
        @classmethod
        def INPUT_TYPES(s): return get_lora_inputs(count)
        RETURN_TYPES = ("UME_LORA_STACK",)
        RETURN_NAMES = ("loras",)
        FUNCTION = "process"
        CATEGORY = "UmeAiRT/Loaders/LoRA"
        def process(self, loras=None, **kwargs): return process_lora_stack(loras, **kwargs)
    _LoraBlock.__name__ = f"UmeAiRT_LoraBlock_{count}"
    _LoraBlock.__qualname__ = f"UmeAiRT_LoraBlock_{count}"
    _LoraBlock.__doc__ = f"A Node to select and stack up to {count} LoRA model(s) with their strengths."
    return _LoraBlock

UmeAiRT_LoraBlock_1  = _make_lora_block_class(1)
UmeAiRT_LoraBlock_3  = _make_lora_block_class(3)
UmeAiRT_LoraBlock_5  = _make_lora_block_class(5)
UmeAiRT_LoraBlock_10 = _make_lora_block_class(10)


# --- ControlNet Blocks ---

class UmeAiRT_ControlNetImageApply:
    """Injects ControlNet configuration into an image bundle.

    Basic mode shows only strength. Advanced inputs expose start/end percent
    and an optional override control image.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input Image Bundle."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"tooltip": "Select ControlNet model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly the ControlNet guides the image. Start with 1.0 and lower if the effect is too strong."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "advanced": True, "tooltip": "When the ControlNet starts influencing (0.0 = from the beginning). Raise to let the AI establish composition first."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "advanced": True, "tooltip": "When the ControlNet stops influencing (1.0 = until the very end). Lower to let the AI refine details freely."}),
            },
            "optional": {
                "optional_control_image": ("IMAGE", {"advanced": True, "tooltip": "Optional: Override control image."}),
            }
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "UmeAiRT/Image"

    def apply_controlnet(self, image_bundle, control_net_name: str, strength: float, start_percent: float = 0.0, end_percent: float = 1.0, optional_control_image: Optional[Any] = None):
        import copy
        new_bundle = copy.copy(image_bundle)
        cnet_stack = list(new_bundle.controlnets) if new_bundle.controlnets else []

        if control_net_name != "None":
            control_use_image = optional_control_image if optional_control_image is not None else new_bundle.image

            if control_use_image is None:
                raise ValueError("ControlNet Image Apply: No Image found in bundle and no optional image provided.")

            cnet_stack.append((control_net_name, control_use_image, strength, start_percent, end_percent))

        new_bundle.controlnets = cnet_stack

        return (new_bundle,)



# --- Parameter Blocks ---


class UmeAiRT_GenerationSettings:
    """Standalone settings node — outputs a dict of generation parameters."""
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
    CATEGORY = "UmeAiRT/Inputs"

    def process(self, width: int, height: int, steps: int, cfg: float, sampler_name: str, scheduler: str, seed: int):
        return (UmeSettings(width=width, height=height, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, seed=seed),)



# --- Image Blocks ---

class UmeAiRT_BlockImageLoader(comfy_nodes.LoadImage):
    """Image loader formatted as a Block.

    Outputs a unified UME_IMAGE bundle plus raw IMAGE and MASK tensors.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "Select an image file to load from disk."}),
            },
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "load_block_image"
    CATEGORY = "UmeAiRT/Image"

    def load_block_image(self, image: str):
        """Loads the specified image file and wraps it in an UmeImage dataclass."""
        out = super().load_image(image)
        img, mask = out[0], out[1]
        image_bundle = UmeImage(image=img, mask=mask, mode="img2img", denoise=0.75)
        return (image_bundle,)

def process_image_core(image_bundle, mode: str, denoise: float = 0.75, auto_resize: bool = False, mask_blur: int = 0, 
                      padding_left: int = 0, padding_top: int = 0, padding_right: int = 0, padding_bottom: int = 0):
    image = image_bundle.image
    mask = image_bundle.mask
    
    if image is None: raise ValueError("Block Image Process: Bundle has no image.")

    B, H, W, C = image.shape
    final_image, final_mask = image, mask

    if mode == "outpaint":
         pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
         final_image, final_mask = apply_outpaint_padding(
             final_image, final_mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40
         )

    if (mode == "inpaint" or mode == "outpaint") and final_mask is not None and mask_blur > 0:
         if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
         else: m = final_mask
         k = mask_blur
         if k % 2 == 0: k += 1
         m = TF.gaussian_blur(m, kernel_size=k)
         final_mask = m.squeeze(0).squeeze(0) if len(final_mask.shape) == 2 else m

    final_mode = "inpaint" if mode in ["inpaint", "outpaint"] else "img2img"
    if mode == "img2img": final_mask = None

    return (UmeImage(image=final_image, mask=final_mask, mode=final_mode, denoise=denoise, auto_resize=auto_resize),)


class UmeAiRT_BlockImageProcess:
    """Structural pre-processor for UME_IMAGE bundles in Block-based workflows.

    Handles cropping, padding (Outpaint mapping), and conditional context tagging.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image. 1.0 = completely new image, 0.5 = keeps half the original detail."}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img", "tooltip": "How to process the image: img2img (transform), inpaint (fill masked area), or outpaint (extend edges)."}),
            },
            "optional": {
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
                "mask_blur": ("INT", {"default": 10, "tooltip": "Softens the edge of the inpaint mask for smoother blending. Higher = softer transitions."}),
                "padding_left": ("INT", {"default": 0, "tooltip": "Pixels to add on the left side when using outpaint mode."}), "padding_top": ("INT", {"default": 0}),
                "padding_right": ("INT", {"default": 0, "tooltip": "Pixels to add on the right side when using outpaint mode."}), "padding_bottom": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Image"

    def process_image(self, image_bundle, denoise: float = 0.75, mode: str = "img2img", auto_resize: bool = False, mask_blur: int = 0, 
                      padding_left: int = 0, padding_top: int = 0, padding_right: int = 0, padding_bottom: int = 0):
        return process_image_core(image_bundle, mode=mode, denoise=denoise, auto_resize=auto_resize, mask_blur=mask_blur, 
                                  padding_left=padding_left, padding_top=padding_top, padding_right=padding_right, padding_bottom=padding_bottom)


class UmeAiRT_ImageProcess_Img2Img:
    """Pre-processor for Img2Img workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image."}),
            },
            "optional": {
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"

    def process(self, image_bundle, denoise=0.75, auto_resize=False):
        return process_image_core(image_bundle, mode="img2img", denoise=denoise, auto_resize=auto_resize)


class UmeAiRT_ImageProcess_Inpaint:
    """Pre-processor for Inpaint workflows."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image inside the mask."}),
            },
            "optional": {
                "mask_blur": ("INT", {"default": 10, "tooltip": "Softens the edge of the inpaint mask for smoother blending."}),
                "auto_resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Settings", "label_off": "Keep Original", "tooltip": "Automatically resize the source image to match the width/height from Generation Settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"

    def process(self, image_bundle, denoise=0.75, mask_blur=10, auto_resize=False):
        return process_image_core(image_bundle, mode="inpaint", denoise=denoise, mask_blur=mask_blur, auto_resize=auto_resize)


class UmeAiRT_ImageProcess_Outpaint:
    """Outpaint configurator — tags the image bundle with target dimensions.

    Does NOT modify the image. The KSampler handles the actual resize,
    padding, mask generation, and blurring at execution time.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image in the padded areas."}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64, "tooltip": "Desired final width of the outpainted image."}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64, "tooltip": "Desired final height of the outpainted image."}),
            },
            "optional": {
                "horizontal_align": (["center", "left", "right"], {"default": "center", "advanced": True, "tooltip": "Where to place the source image horizontally within the target canvas."}),
                "vertical_align": (["center", "top", "bottom"], {"default": "center", "advanced": True, "tooltip": "Where to place the source image vertically within the target canvas."}),
                "mask_blur": ("INT", {"default": 10, "advanced": True, "tooltip": "Softens the edge of the outpaint mask for smoother blending."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Image"

    def process(self, image_bundle, denoise=0.75, target_width=1024, target_height=1024,
                horizontal_align="center", vertical_align="center", mask_blur=10):
        import copy
        new_bundle = copy.copy(image_bundle)
        new_bundle.mode = "outpaint"
        new_bundle.denoise = denoise
        new_bundle.auto_resize = False
        new_bundle.outpaint_target_w = target_width
        new_bundle.outpaint_target_h = target_height
        new_bundle.outpaint_h_align = horizontal_align
        new_bundle.outpaint_v_align = vertical_align
        new_bundle.outpaint_mask_blur = mask_blur
        return (new_bundle,)

class UmeAiRT_Positive_Input:
    """Multiline text editor for the positive prompt. Outputs a STRING."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive prompt."}),
            }
        }

    RETURN_TYPES = ("POSITIVE",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "pass_through"
    CATEGORY = "UmeAiRT/Inputs"

    def pass_through(self, positive):
        return (positive,)


class UmeAiRT_Negative_Input:
    """Multiline text editor for the negative prompt. Outputs a STRING."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative": ("STRING", {"default": "text, watermark", "multiline": True, "dynamicPrompts": True, "tooltip": "Negative prompt."}),
            }
        }

    RETURN_TYPES = ("NEGATIVE",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "pass_through"
    CATEGORY = "UmeAiRT/Inputs"

    def pass_through(self, negative):
        return (negative,)
