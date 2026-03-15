"""
UmeAiRT Toolkit - Common Shared State & Utils
---------------------------------------------
Core utilities, typed bundle definitions, and the GenerationContext pipeline object.
"""

import copy
import torch
from typing import TypedDict, Any, Optional, List, Tuple
from .logger import log_node


# --- Typed Bundle Definitions ---

class UmeBundle(TypedDict):
    """Type contract for UME_BUNDLE — produced by all Loader nodes."""
    model: Any
    clip: Any
    vae: Any
    model_name: str


class UmeSettings(TypedDict):
    """Type contract for UME_SETTINGS — produced by GenerationSettings."""
    width: int
    height: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    seed: int


class UmeImage(TypedDict, total=False):
    """Type contract for UME_IMAGE — produced by Image Loader/Process nodes."""
    image: Any
    mask: Any
    mode: str
    denoise: float
    auto_resize: bool
    controlnets: List[Tuple]


# --- GenerationContext (UME_PIPELINE) ---

class GenerationContext:
    """Encapsulates all state for a single generation pipeline.

    Created by the BlockSampler, this object carries models, settings,
    prompts, and the generated image through the post-processing chain.
    """
    def __init__(self):
        # Models
        self.model = None
        self.clip = None
        self.vae = None
        self.model_name = ""

        # Settings
        self.width = 1024
        self.height = 1024
        self.steps = 20
        self.cfg = 8.0
        self.sampler_name = "euler"
        self.scheduler = "normal"
        self.seed = 0
        self.denoise = 1.0

        # Prompts
        self.positive_prompt = ""
        self.negative_prompt = ""

        # Generated output
        self.image = None
        self.latent = None

        # Extras
        self.loras = []
        self.controlnets = []
        self.source_image = None
        self.source_mask = None

    def clone(self):
        """Create an independent copy for branched workflows."""
        ctx = copy.copy(self)
        ctx.loras = list(self.loras)
        ctx.controlnets = list(self.controlnets)
        return ctx

    def is_ready(self):
        """Validates that minimum required data is set for sampling."""
        return self.model is not None and self.vae is not None and self.clip is not None


# --- Tensor Utilities ---

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


def encode_prompts(clip, pos_text, neg_text):
    """Encode positive and negative text prompts into conditioning tensors.

    Centralizes the CLIP tokenize → encode → format pattern used across
    all pipeline-aware nodes (FaceDetailer, Upscaler, Detailer Daemon, etc.).

    Args:
        clip: The loaded CLIP model from ComfyUI.
        pos_text (str): The positive prompt text.
        neg_text (str): The negative prompt text.

    Returns:
        tuple: (positive_cond, negative_cond) ready for KSampler.
    """
    tokens = clip.tokenize(pos_text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    positive = [[cond, {"pooled_output": pooled}]]

    tokens = clip.tokenize(neg_text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    negative = [[cond, {"pooled_output": pooled}]]
    return positive, negative


def apply_outpaint_padding(image, mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40):
    """Apply outpaint padding to an image and generate the corresponding mask.

    Stretches edge pixels outward using 'replicate' padding, creates a mask marking
    the padded regions (with overlap into the original area), and applies Gaussian
    feathering for smooth transitions.

    Args:
        image (torch.Tensor): Input image tensor [B, H, W, C].
        mask (torch.Tensor or None): Optional existing mask [B, H, W].
        pad_l (int): Left padding in pixels.
        pad_t (int): Top padding in pixels.
        pad_r (int): Right padding in pixels.
        pad_b (int): Bottom padding in pixels.
        overlap (int, optional): Overlap into original image in pixels. Defaults to 8.
        feathering (int, optional): Gaussian blur kernel size for mask feathering. Defaults to 40.

    Returns:
        tuple: (padded_image, padded_mask) with the same tensor formats.
    """
    if pad_l <= 0 and pad_t <= 0 and pad_r <= 0 and pad_b <= 0:
        return image, mask

    B, H, W, C = image.shape

    # Pad image using replicate mode
    img_p = image.permute(0, 3, 1, 2)
    img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
    final_image = img_padded.permute(0, 2, 3, 1)

    # Build outpaint mask
    new_h = H + pad_t + pad_b
    new_w = W + pad_l + pad_r
    new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)

    if mask is not None:
        if len(mask.shape) == 2:
            m_in = mask.unsqueeze(0)
        else:
            m_in = mask
        m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
        if len(mask.shape) == 2:
            new_mask = m_padded.squeeze(0)
        else:
            new_mask = m_padded

    # Mark padded regions (with overlap into original area)
    if pad_t > 0: new_mask[:, :pad_t + overlap, :] = 1.0
    if pad_b > 0: new_mask[:, -(pad_b + overlap):, :] = 1.0
    if pad_l > 0: new_mask[:, :, :pad_l + overlap] = 1.0
    if pad_r > 0: new_mask[:, :, -(pad_r + overlap):] = 1.0

    # Feathering (Gaussian blur)
    if feathering > 0:
        import torchvision.transforms.functional as TF
        k = feathering
        if k % 2 == 0: k += 1
        sig = float(k) / 3.0
        if len(new_mask.shape) == 2:
            m_b = new_mask.unsqueeze(0).unsqueeze(0)
        else:
            m_b = new_mask.unsqueeze(1)
        m_b = TF.gaussian_blur(m_b, kernel_size=k, sigma=sig)
        if len(new_mask.shape) == 2:
            new_mask = m_b.squeeze(0).squeeze(0)
        else:
            new_mask = m_b.squeeze(1)

    return final_image, new_mask

