import torch
import os
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, resize_tensor, apply_outpaint_padding, log_node
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
        CATEGORY = "UmeAiRT/Block/LoRA"
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

class UmeAiRT_ControlNetImageApply_Advanced:
    """Injects deeply parameterized ControlNet configuration into an image bundle.
    
    Allows explicitly separating the target image mapped from the control image (optional).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE", {"tooltip": "Input Image Bundle."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"tooltip": "Select ControlNet model."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly the ControlNet guides the image. Start with 1.0 and lower if the effect is too strong."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "When the ControlNet starts influencing (0.0 = from the beginning). Raise to let the AI establish composition first."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "When the ControlNet stops influencing (1.0 = until the very end). Lower to let the AI refine details freely."}),
            },
            "optional": {
                "optional_control_image": ("IMAGE", {"tooltip": "Optional: Override control image."}), 
            }
        }

    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "UmeAiRT/Block/ControlNet"

    def apply_controlnet(self, image_bundle: Dict[str, Any], control_net_name: str, strength: float, start_percent: float, end_percent: float, optional_control_image: Optional[Any] = None) -> Tuple[Dict[str, Any]]:
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

        return (new_bundle,)

class UmeAiRT_ControlNetImageApply_Simple(UmeAiRT_ControlNetImageApply_Advanced):
    """Injects a simplified ControlNet configuration into an image bundle."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "display": "slider", "tooltip": "How strongly the ControlNet guides the image. Start with 1.0 and lower if the effect is too strong."}),
            }
        }
    def apply_controlnet(self, image_bundle: Dict[str, Any], control_net_name: str, strength: float) -> Tuple[Dict[str, Any]]:
        """Funnels simple parameters down to the advanced method safely.

        Args:
            image_bundle (dict): The target UME_IMAGE bundle.
            control_net_name (str): Selected model filename.
            strength (float): ControlNet force multiplier.

        Returns:
            tuple: A tuple containing the updated image bundle.
        """
        return super().apply_controlnet(image_bundle, control_net_name, strength, 0.0, 1.0, None)

class UmeAiRT_ControlNetImageProcess:
    """Pre-processes image data (resize, mask blurring, inversion) before embedding ControlNets.
    
    Provides specialized routing for specific control scenarios.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_bundle": ("UME_IMAGE",),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much the AI changes the image. 1.0 = completely new image, 0.5 = keeps half the original detail."}),
                "mode": (["img2img", "txt2img"], {"default": "img2img", "tooltip": "How to process the image: img2img (transform), inpaint (fill masked area), or outpaint (extend edges)."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "How strongly the ControlNet guides the image. Start with 1.0 and lower if the effect is too strong."}),
            },
            "optional": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "Generation pipeline (optional, for resize dimensions)."}),
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Automatically resize the image to match your generation width/height settings."}),
            }
        }
    RETURN_TYPES = ("UME_IMAGE",)
    RETURN_NAMES = ("image_bundle",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Block/ControlNet"

    def process(self, image_bundle: Dict[str, Any], denoise: float, mode: str, control_net_name: str, strength: float, gen_pipe: Optional[GenerationContext] = None, resize: bool = False) -> Tuple[Dict[str, Any]]:
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
             target_w = gen_pipe.width if gen_pipe else 1024
             target_h = gen_pipe.height if gen_pipe else 1024
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
    CATEGORY = "UmeAiRT/Block/Settings"

    def process(self, width: int, height: int, steps: int, cfg: float, sampler_name: str, scheduler: str, seed: int) -> Tuple[Dict[str, Any]]:
        return ({"width": width, "height": height, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "seed": seed},)



# --- Image Blocks ---

class UmeAiRT_BlockImageLoader(comfy_nodes.LoadImage):
    """Standard image loader formatted as a Block.

    Outputs a unified UME_IMAGE bundle containing the image and its associated mask.
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
    CATEGORY = "UmeAiRT/Block/Image"

    def load_block_image(self, image: str) -> Tuple[Dict[str, Any]]:
        """Loads the specified image file and wraps it in a dictionary.

        Args:
            image (str): The filename to load from the input directory.

        Returns:
            tuple: A tuple containing the `{"image": img, "mask": mask}` bundle.
        """
        out = super().load_image(image)
        img, mask = out[0], out[1]

        image_bundle = {"image": img, "mask": mask, "mode": "img2img", "denoise": 0.75}
        return (image_bundle,)

class UmeAiRT_BlockImageLoader_Advanced(UmeAiRT_BlockImageLoader):
    """Advanced Image Loader providing both bundled and fragmented UI outputs."""
    RETURN_TYPES = ("UME_IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image_bundle", "image", "mask")
    def load_block_image(self, image: str) -> Tuple[Dict[str, Any], Any, Any]:
        """Loads the image and returns both the bundle and the raw tensors.

        Args:
            image (str): The filename.

        Returns:
            tuple: `(bundle_dict, image_tensor, mask_tensor)`
        """
        res = super().load_block_image(image)
        return (res[0], res[0]["image"], res[0]["mask"])

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
    CATEGORY = "UmeAiRT/Block/Image"

    def process_image(self, image_bundle: Dict[str, Any], denoise: float = 0.75, mode: str = "img2img", auto_resize: bool = False, mask_blur: int = 0, 
                      padding_left: int = 0, padding_top: int = 0, padding_right: int = 0, padding_bottom: int = 0) -> Tuple[Dict[str, Any]]:
        """Modifies the image state based on the chosen mode."""
        
        image = image_bundle.get("image")
        mask = image_bundle.get("mask")
        
        if image is None: raise ValueError("Block Image Process: Bundle has no image.")

        if mode == "txt2img":
             log_node("Block Process: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             denoise = 1.0
             mask = None 

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
        if mode == "txt2img": final_mode = "txt2img"
        elif mode == "img2img": final_mask = None

        return ({"image": final_image, "mask": final_mask, "mode": final_mode, "denoise": denoise, "auto_resize": auto_resize},)

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
    CATEGORY = "UmeAiRT/Block/Prompts"

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
    CATEGORY = "UmeAiRT/Block/Prompts"

    def pass_through(self, negative):
        return (negative,)
