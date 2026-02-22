"""
UmeAiRT Toolkit - Common Shared State & Utils
---------------------------------------------
Stores the global dictionary and keys to avoid circular imports.
"""

import torch
from .logger import log_node

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
KEY_IMAGE = "ume_internal_image"
KEY_SOURCE_IMAGE = "ume_internal_source_image"
KEY_SOURCE_MASK = "ume_internal_source_mask"
KEY_CONTROLNETS = "ume_internal_controlnets"

def resize_tensor(tensor, target_h, target_w, interp_mode="bilinear", is_mask=False):
    """Resizes an image or mask tensor to the target dimensions.

    Handles dimension permutations between ComfyUI format (B, H, W, C) 
    and PyTorch format (B, C, H, W) before applying the interpolation.

    Args:
        tensor (torch.Tensor): The input tensor representing an image or a mask.
        target_h (int): The target height in pixels.
        target_w (int): The target width in pixels.
        interp_mode (str, optional): The interpolation mode used by torch.nn.functional.interpolate. Defaults to "bilinear".
        is_mask (bool, optional): If True, treats the input as a mask (B, H, W). Defaults to False.

    Returns:
        torch.Tensor: The resized tensor, returned in its original ComfyUI dimension format.
    """
    if is_mask:
        # Mask: [B, H, W] -> [B, 1, H, W]
        t = tensor.unsqueeze(1)
    else:
        # Image: [B, H, W, C] -> [B, C, H, W]
        t = tensor.permute(0, 3, 1, 2)
    
    t_resized = torch.nn.functional.interpolate(t, size=(target_h, target_w), mode=interp_mode, align_corners=False if interp_mode!="nearest" else None)
    
    if is_mask:
        # [B, 1, H, W] -> [B, H, W] #
        return t_resized.squeeze(1)
    else:
        # [B, C, H, W] -> [B, H, W, C]
        return t_resized.permute(0, 2, 3, 1)
