import torch
import numpy as np
import os
import folder_paths
import comfy.utils
import comfy.sd
import nodes as comfy_nodes
import comfy.samplers
import comfy.sample
from .common import (
    UME_SHARED_STATE, KEY_MODEL, KEY_CLIP, KEY_VAE, KEY_POSITIVE, KEY_NEGATIVE, 
    KEY_LATENT, KEY_SEED, KEY_STEPS, KEY_CFG, KEY_SAMPLER, KEY_SCHEDULER, 
    KEY_DENOISE, KEY_SOURCE_IMAGE, KEY_SOURCE_MASK, KEY_IMAGESIZE, KEY_LORAS, 
    KEY_CONTROLNETS, KEY_IMAGE, log_node
)
from .logger import logger

# Try import internals
try:
    from .seedvr2_adapter import execute_seedvr2
    from .stitching import process_and_stitch
except ImportError:
    pass

try:
    from .facedetailer_core import logic as fd_logic
    from .facedetailer_core import detector
except ImportError:
    pass


# --- Helpers ---

from .optimization_utils import SamplerContext

def _ensure_vram_for_decode():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except:
            pass

def _ensure_vram_for_seedvr2():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Wireless KSampler ---

class UmeAiRT_WirelessKSampler:
    """
    Central Wireless Sampler.
    Fetches all required inputs (Model, VAE, CLIP, Latent, Params) from Wireless State.
    Handles ControlNet application and LoRA loading if present in state.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "signal": ("ANY", {"default": None, "forceInput": True, "tooltip": "Trigger signal (connect anything)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "VAE")
    RETURN_NAMES = ("image", "latent", "vae")
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Wireless/Processors"

    def __init__(self):
         self.cnet_loader = comfy_nodes.ControlNetLoader()
         self.cnet_apply = comfy_nodes.ControlNetApplyAdvanced()
         
         # Local Prompt Cache
         self._last_pos_text = None
         self._last_neg_text = None
         self._last_clip = None
         self._cached_positive = None
         self._cached_negative = None

    def process(self, signal=None):
        # 1. Fetch State
        model = UME_SHARED_STATE.get(KEY_MODEL)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        
        # Params
        steps = UME_SHARED_STATE.get(KEY_STEPS, 20)
        cfg = UME_SHARED_STATE.get(KEY_CFG, 8.0)
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        seed = UME_SHARED_STATE.get(KEY_SEED, 0)
        denoise = UME_SHARED_STATE.get(KEY_DENOISE, 1.0)
        
        # Prompts
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        # 2. Validation
        missing = []
        if not model: missing.append("MODEL")
        if not clip: missing.append("CLIP")
        if not vae: missing.append("VAE")
        
        if missing:
             raise ValueError(f"Wireless KSampler: Missing {', '.join(missing)}. Check your Wireless Loaders.")

        # 3. Latent & Mode Logic
        # Priority: Wireless Latent (if set by some node) > Image Input (Img2Img) > Empty (Txt2Img)
        
        latent_image = UME_SHARED_STATE.get(KEY_LATENT)
        
        source_image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        source_mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK) # Only present if Inpaint mode
        
        mode = "Txt2Img"
        
        if latent_image is None:
            # Img2Img / Inpaint Check
            mode_from_state = UME_SHARED_STATE.get(KEY_MODE, "img2img")  # Fallback logic

            if source_image is not None and source_mask is not None:
                 # For inpaint/outpaint, ALWAYS encode and use the mask, even if denoise >= 1.0
                 t = vae.encode(source_image[:,:,:,:3]) # Drop alpha if 4 channels
                 latent_image = {"samples": t, "noise_mask": source_mask}
                 mode = "Inpaint/Outpaint"
            elif source_image is not None and denoise < 1.0:
                 # Standard Img2Img encoded only if denoise < 1.0
                 t = vae.encode(source_image[:,:,:,:3])
                 latent_image = {"samples": t}
                 mode = "Img2Img"
            else:
                 # Pure Txt2Img
                 size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
                 w = int(size.get("width", 1024))
                 h = int(size.get("height", 1024))
                 batch_size = 1
                 
                 # Empty Latent Logic
                 l = torch.zeros([batch_size, 4, h // 8, w // 8], device="cpu")
                 latent_image = {"samples": l}
                 mode = "Txt2Img"
                 denoise = 1.0
        
        # Just in case KEY_MODE wasn't exactly set
        if mode != "Txt2Img":
             pass

        # 4. Encoding Prompts
        if self._last_pos_text == pos_text and self._last_neg_text == neg_text and self._last_clip is clip:
             positive = self._cached_positive
             negative = self._cached_negative
             log_node("Wireless Sampler: Using cached Prompts (Fast Start)", color="GREEN")
        else:
             log_node("Wireless Sampler: Encoding Prompts...")
             tokens = clip.tokenize(pos_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             positive = [[cond, {"pooled_output": pooled}]]

             tokens = clip.tokenize(neg_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             negative = [[cond, {"pooled_output": pooled}]]
             
             self._last_pos_text = pos_text
             self._last_neg_text = neg_text
             self._last_clip = clip
             self._cached_positive = positive
             self._cached_negative = negative

        # 5. Apply ControlNets (if in State)
        cnets = UME_SHARED_STATE.get(KEY_CONTROLNETS, [])
        if cnets:
            for cnet_def in cnets:
                # (name, image, strength, start, end)
                c_name, c_image, c_str, c_start, c_end = cnet_def
                if c_name != "None" and c_image is not None:
                    try:
                        c_model = self.cnet_loader.load_controlnet(c_name)[0]
                        positive, negative = self.cnet_apply.apply_controlnet(
                            positive, negative, c_model, c_image, c_str, c_start, c_end
                        )
                        log_node(f"Applied Wireless ControlNet: {c_name}", color="BLUE")
                    except Exception as e:
                        log_node(f"Wireless ControlNet Error ({c_name}): {e}", color="RED")

        # 6. Sample
        log_node(f"Wireless Sampler: {mode} | Steps: {steps} | CFG: {cfg} | Denoise: {denoise}")
        
        # Sampler Context Wrapper
        try:
             from .optimization_utils import warmup_vae
             warmup_vae(vae)
             
             with SamplerContext():
                 result_latent = comfy_nodes.KSampler().sample(
                     model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
                 )[0]
        except Exception as e:
             raise RuntimeError(f"Sampling Failed: {e}")

        # 7. Decode
        log_node("Wireless Sampler: Decoding VAE (First run may pause for VRAM loading & compilation)...")
        _ensure_vram_for_decode()
        image = comfy_nodes.VAEDecode().decode(vae, result_latent)[0]

        # 8. Update State (optional output)
        UME_SHARED_STATE[KEY_IMAGE] = image
        UME_SHARED_STATE[KEY_LATENT] = result_latent # For chained samplers?

        return (image, result_latent, vae)


# --- Upscalers ---

class UmeAiRT_WirelessUltimateUpscale_Base:
    """
    Base class for Wireless Ultimate Upscale nodes.
    Contains logic for prompt encoding and node execution.
    """
    def encode_prompts(self, clip, pos_text, neg_text):
        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]
        return positive, negative

    def get_usdu_node(self):
        import sys
        import os
        # logic_nodes.py is in modules/, so toolkit root is one level up
        usdu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usdu_core")
        if usdu_path not in sys.path:
            sys.path.append(usdu_path)
        try:
            import usdu_main
            return usdu_main.UltimateSDUpscale()
        except ImportError as e:
            raise ImportError(f"UmeAiRT: Could not load bundled UltimateSDUpscale node from usdu_core. Error: {e}")

class UmeAiRT_WirelessUltimateUpscale(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough"}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
            },
            "optional":{}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def upscale(self, image, enabled, model, upscale_by):
        log_node(f"UltimateSDUpscale (Simple): Processing | Ratio: x{upscale_by} | Model: {model} | Denoise: 0.35")
        if not enabled:
            return (image,)

        # Default optimal values for the simple node
        denoise = 0.35 
        clean_prompt = True
        mode_type = "Linear"
        tile_padding = 32
        # Fetch Logic
        sd_model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
        cfg = 1.0 # USDU often works best with low CFG
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        
        if not sd_model or not vae or not clip:
            raise ValueError("Wireless USDU: Missing Model/VAE/CLIP.")

        # Prompts
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))
        
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)

        # Load Upscale Model
        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        usdu_node = self.get_usdu_node()
        steps = max(5, steps // 4) # Reduced steps for upscale
        
        size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
        if not isinstance(size, dict): size = {"width": 1024, "height": 1024}
        tile_width = int(size.get("width", 1024))
        tile_height = int(size.get("height", 1024))

        with SamplerContext():
             res = usdu_node.upscale(
                 image=image, model=sd_model, positive=positive, negative=negative, vae=vae,
                 upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
                 sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
                 upscale_model=upscale_model, mode_type=mode_type,
                 tile_width=tile_width, tile_height=tile_height, mask_blur=16, tile_padding=tile_padding,
                 seam_fix_mode="None", seam_fix_denoise=1.0,
                 seam_fix_mask_blur=8, seam_fix_width=64, seam_fix_padding=16,
                 force_uniform_tiles=True, tiled_decode=False,
                 suppress_preview=True,
             )
        
        return res

class UmeAiRT_WirelessUltimateUpscale_Advanced(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        seam_fix_modes = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
        return {
             "required": {
                "image": ("IMAGE",),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            },
            "optional":{
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Global Prompt"}),
                "mode_type": (usdu_modes, {"default": "Linear"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128}),
                "seam_fix_mode": (seam_fix_modes, {"default": "None"}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 512}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def upscale(self, image, model, upscale_by, denoise, clean_prompt=True, mode_type="Linear", tile_width=512, tile_height=512, mask_blur=8, tile_padding=32, seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_width=64, seam_fix_mask_blur=8, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False):
        
        log_node(f"UltimateSDUpscale (Advanced): Processing | Ratio: x{upscale_by} | Model: {model} | Denoise: {denoise}")

        sd_model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
        steps = max(5, steps // 4)
        
        cfg = 1.0
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        
        if not sd_model or not vae or not clip:
            raise ValueError("Wireless USDU Advanced: Missing Model/VAE/CLIP.")

        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)

        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        usdu_node = self.get_usdu_node()
        
        with SamplerContext():
             res = usdu_node.upscale(
                 image=image, model=sd_model, positive=positive, negative=negative, vae=vae,
                 upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
                 sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
                 upscale_model=upscale_model, mode_type=mode_type,
                 tile_width=tile_width, tile_height=tile_height, mask_blur=mask_blur, tile_padding=tile_padding,
                 seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
                 seam_fix_mask_blur=seam_fix_mask_blur, seam_fix_width=seam_fix_width, seam_fix_padding=seam_fix_padding,
                 force_uniform_tiles=force_uniform_tiles, tiled_decode=tiled_decode,
                 suppress_preview=True,
             )
        
        return res

# ---------------------------------------------------------------------------
#  VRAM Management for SeedVR2
# ---------------------------------------------------------------------------
SEEDVR2_VRAM_REQUIRED = 6 * 1024 * 1024 * 1024  # 6 GB
DECODE_VRAM_REQUIRED = 1.5 * 1024 * 1024 * 1024 # 1.5 GB

def _ensure_vram_for_decode():
    """Ensure sufficient VRAM for VAE Decode."""
    import gc
    import comfy.model_management as mm
    device = mm.get_torch_device()
    free_before = mm.get_free_memory(device)
    if free_before >= DECODE_VRAM_REQUIRED:
        log_node(f"Decode VRAM Check: Safe ({free_before / (1024**3):.2f} GB available) -> skipping cleanup")
        return
    log_node(f"Decode VRAM Check: WARNING | Low VRAM ({free_before / (1024**3):.2f} GB) -> clearing cache...", color="ORANGE")
    mm.soft_empty_cache()
    if mm.get_free_memory(device) < DECODE_VRAM_REQUIRED:
         mm.free_memory(DECODE_VRAM_REQUIRED, device)
         gc.collect()
         log_node("Decode VRAM Check: Models unloaded to free VRAM for Decode.")

def _ensure_vram_for_seedvr2():
    """Check available VRAM and unload cached models if necessary."""
    import gc
    import comfy.model_management as mm
    device = mm.get_torch_device()
    free_before = mm.get_free_memory(device)
    free_gb = free_before / (1024 ** 3)
    log_node(f"SeedVR2 VRAM Check: {free_gb:.2f} GB free, {SEEDVR2_VRAM_REQUIRED / (1024**3):.0f} GB required")
    if free_before >= SEEDVR2_VRAM_REQUIRED:
        log_node("SeedVR2 VRAM Check: OK -> skipping cleanup")
        return
    log_node("SeedVR2 VRAM Check: WARNING | Insufficient VRAM -> unloading cached models...", color="ORANGE")
    mm.free_memory(SEEDVR2_VRAM_REQUIRED, device)
    gc.collect()
    mm.soft_empty_cache()
    free_after = mm.get_free_memory(device)
    freed_mb = (free_after - free_before) / (1024 ** 2)
    log_node(f"SeedVR2 VRAM Check: Cleanup done -> freed {freed_mb:.0f} MB -> now {free_after / (1024**3):.2f} GB free")

class UmeAiRT_WirelessSeedVR2Upscale:
    @classmethod
    def INPUT_TYPES(s):
        # Legacy Discovery Logic
        KNOWN_DIT_MODELS = [
            "seedvr2_ema_3b-Q4_K_M.gguf",
            "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf",
            "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
        ]
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"

        try:
             # Attempt to discover extra models using SeedVR2 logic
             # We need to import relatively from root package
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception:
             # Fallback to hardcoded list if adapter/module not found
             dit_models = KNOWN_DIT_MODELS
        
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough"}),
                "model": (dit_models, {"default": default_dit, "tooltip": "DiT model for SeedVR2 upscaling."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "display": "slider"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    @staticmethod
    def _build_configs(model_name: str):
        """Build dit_config and vae_config dicts (same format as loader nodes)."""
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dit_config = {
            "model": model_name,
            "device": device,
            "offload_device": "cpu",
            "cache_model": False,
            "blocks_to_swap": 0,
            "swap_io_components": False,
            "attention_mode": "sdpa",
            "torch_compile_args": None,
            "node_id": None,
        }

        vae_config = {
            "model": "ema_vae_fp16.safetensors",
            "device": device,
            "offload_device": "cpu",
            "cache_model": False,
            "encode_tiled": False,
            "encode_tile_size": 1024,
            "encode_tile_overlap": 128,
            "decode_tiled": False,
            "decode_tile_size": 1024,
            "decode_tile_overlap": 128,
            "tile_debug": "false",
            "torch_compile_args": None,
            "node_id": None,
        }

        return dit_config, vae_config

    def upscale(self, image, enabled, model, upscale_by):
        if not enabled:
            return (image,)

        # Legacy Imports
        try:
            from ..seedvr2_core.image_utils import tensor_to_pil, pil_to_tensor
            from ..seedvr2_core.tiling import generate_tiles
            from ..seedvr2_core.stitching import process_and_stitch
        except ImportError:
             raise ImportError("SeedVR2 Core modules not found in '../seedvr2_core'. verify installation.")

        # Wireless seed
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 100)) % (2**32) 

        # Build model configs internally
        dit_config, vae_config = self._build_configs(model)

        log_node(f"SeedVR2 Upscale: Processing | Ratio: x{upscale_by} | Model: {model} | Seed: {seed}")
        _ensure_vram_for_seedvr2()

        # VRAM Consultation & Warning System
        import comfy.model_management as mm
        device = mm.get_torch_device()
        free_vram_gb = mm.get_free_memory(device) / (1024**3)
        total_vram_gb = mm.get_total_memory(device) / (1024**3)

        model_l = model.lower()
        if "7b" in model_l:
            if "q4" in model_l: m_size_gb = 4.8 
            elif "fp16" in model_l and "mixed" not in model_l: m_size_gb = 16.5
            else: m_size_gb = 8.5 # fp8 / mixed
        else: # 3B
            if "q4" in model_l: m_size_gb = 2.0
            elif "q8" in model_l: m_size_gb = 3.7
            elif "fp16" in model_l: m_size_gb = 6.8
            else: m_size_gb = 3.4 # fp8

        overhead_gb = 3.5 # Generous margin for VAE, Activations, and OS
        req_vram = m_size_gb + overhead_gb

        if total_vram_gb < req_vram:
            if total_vram_gb < m_size_gb:
                log_node(f"SeedVR2 Upscale: CRITICAL | Model '{model}' (~{m_size_gb:.1f}GB) is LARGER than your total VRAM ({total_vram_gb:.1f}GB)! Expect EXTREME OOM slowdown.", color="RED")
                if total_vram_gb <= 8.5:
                    log_node("SeedVR2 Upscale: ADVICE | For 8GB GPUs, please use 'seedvr2_ema_3b_fp8_e4m3fn.safetensors' (3.4GB) or ideally 'seedvr2_ema_3b-Q4_K_M.gguf' (2GB).")
            else:
                log_node(f"SeedVR2 Upscale: WARNING | VRAM is very tight. Model (~{m_size_gb:.1f}GB) + Process Overhead (~{overhead_gb}GB) > Total VRAM ({total_vram_gb:.1f}GB). This will cause slow Shared RAM swap.", color="ORANGE")
                if total_vram_gb <= 8.5:
                    if "7b" in model_l:
                        log_node("SeedVR2 Upscale: ADVICE | 7B models are too heavy for an 8GB GPU. Switch to 'seedvr2_ema_3b_fp8_e4m3fn.safetensors' or 'seedvr2_ema_3b-Q4_K_M.gguf'.")
                    elif m_size_gb > 3.0:
                        log_node("SeedVR2 Upscale: ADVICE | To avoid the slow RAM swap on 8GB GPUs, drop down to 'seedvr2_ema_3b-Q4_K_M.gguf' (2GB).")
                elif total_vram_gb <= 12.5:
                    log_node("SeedVR2 Upscale: ADVICE | For 12GB GPUs, 'seedvr2_ema_7b-Q4_K_M.gguf' (~4.8GB) is the sweet spot for 7B models.")
            
            if total_vram_gb <= 8.5:
                log_node("SeedVR2 Upscale: TIP | If SeedVR2 is still too slow due to VRAM limits, try using the 'Wireless UltimateSDUpscale' node instead. It is much lighter on memory!")
        else:
             log_node(f"SeedVR2 Upscale: VRAM Check OK | {total_vram_gb:.1f}GB total VRAM is plenty for model '{model}' (~{m_size_gb:.1f}GB).", color="GREEN")

        # Best-practice defaults restored for quality
        tile_width = 512
        tile_height = 512
        mask_blur = 0
        tile_padding = 32
        tile_upscale_resolution = 1024
        tiling_strategy = "Chess"
        anti_aliasing_strength = 0.0
        blending_method = "auto"
        color_correction = "lab"

        pil_image = tensor_to_pil(image)
        upscale_factor = upscale_by
        output_width = int(pil_image.width * upscale_factor)
        output_height = int(pil_image.height * upscale_factor)

        main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)

        output_image = process_and_stitch(
            tiles=main_tiles,
            width=output_width, height=output_height,
            dit_config=dit_config, vae_config=vae_config,
            seed=seed,
            tile_upscale_resolution=tile_upscale_resolution,
            upscale_factor=upscale_factor,
            mask_blur=mask_blur,
            progress=None,
            original_image=pil_image,
            anti_aliasing_strength=anti_aliasing_strength,
            blending_method=blending_method,
            color_correction=color_correction,
        )

        # Cleanup VRAM after massive DiT model use
        import gc
        import comfy.model_management as mm
        log_node("SeedVR2 Upscale: Finished | VRAM cleared", color="GREEN")
        mm.soft_empty_cache()
        gc.collect()

        return (pil_to_tensor(output_image),)
        

class UmeAiRT_WirelessSeedVR2Upscale_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        # Rediscover models for all-in-one experience
        KNOWN_DIT_MODELS = [
            "seedvr2_ema_3b-Q4_K_M.gguf", "seedvr2_ema_3b-Q8_0.gguf", "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors", "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_sharp-Q4_K_M.gguf", "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
        ]
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        try:
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception:
             dit_models = KNOWN_DIT_MODELS

        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough"}),
                "model": (dit_models, {"default": default_dit}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "display": "slider"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tiling_strategy": (["Chess", "Linear"], {"default": "Chess"}),
                "anti_aliasing_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blending_method": (["auto", "multiband", "bilateral", "content_aware", "linear", "simple"], {"default": "auto"}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def upscale(self, image, enabled, model, upscale_by,
                tile_width, tile_height, mask_blur, tile_padding,
                tile_upscale_resolution, tiling_strategy,
                anti_aliasing_strength, blending_method, color_correction):
        if not enabled:
            return (image,)

        try:
            from ..seedvr2_core.image_utils import tensor_to_pil, pil_to_tensor
            from ..seedvr2_core.tiling import generate_tiles
            from ..seedvr2_core.stitching import process_and_stitch
        except ImportError:
             raise ImportError("SeedVR2 Core modules not found. verify installation.")

        seed = int(UME_SHARED_STATE.get(KEY_SEED, 100)) % (2**32) 
        
        # Build model configs internally (Advanced node in refactor is also all-in-one)
        dit_config, vae_config = UmeAiRT_WirelessSeedVR2Upscale._build_configs(model)

        log_node(f"SeedVR2 Upscale: Processing | Ratio: x{upscale_by} | Model: {model} | Seed: {seed}")
        _ensure_vram_for_seedvr2()

        # VRAM Consultation & Warning System
        import comfy.model_management as mm
        device = mm.get_torch_device()
        free_vram_gb = mm.get_free_memory(device) / (1024**3)
        total_vram_gb = mm.get_total_memory(device) / (1024**3)

        model_l = model.lower()
        if "7b" in model_l:
            if "q4" in model_l: m_size_gb = 4.8 
            elif "fp16" in model_l and "mixed" not in model_l: m_size_gb = 16.5
            else: m_size_gb = 8.5 # fp8 / mixed
        else: # 3B
            if "q4" in model_l: m_size_gb = 2.0
            elif "q8" in model_l: m_size_gb = 3.7
            elif "fp16" in model_l: m_size_gb = 6.8
            else: m_size_gb = 3.4 # fp8

        overhead_gb = 3.5 # Generous margin for VAE, Activations, and OS
        req_vram = m_size_gb + overhead_gb

        if total_vram_gb < req_vram:
            if total_vram_gb < m_size_gb:
                log_node(f"SeedVR2 Upscale: CRITICAL | Model '{model}' (~{m_size_gb:.1f}GB) is LARGER than your total VRAM ({total_vram_gb:.1f}GB)! Expect EXTREME OOM slowdown.", color="RED")
                if total_vram_gb <= 8.5:
                    log_node("SeedVR2 Upscale: ADVICE | For 8GB GPUs, please use 'seedvr2_ema_3b_fp8_e4m3fn.safetensors' (3.4GB) or ideally 'seedvr2_ema_3b-Q4_K_M.gguf' (2GB).")
            else:
                log_node(f"SeedVR2 Upscale: WARNING | VRAM is very tight. Model (~{m_size_gb:.1f}GB) + Process Overhead (~{overhead_gb}GB) > Total VRAM ({total_vram_gb:.1f}GB). This will cause slow Shared RAM swap.", color="ORANGE")
                if total_vram_gb <= 8.5:
                    if "7b" in model_l:
                        log_node("SeedVR2 Upscale: ADVICE | 7B models are too heavy for an 8GB GPU. Switch to 'seedvr2_ema_3b_fp8_e4m3fn.safetensors' or 'seedvr2_ema_3b-Q4_K_M.gguf'.")
                    elif m_size_gb > 3.0:
                        log_node("SeedVR2 Upscale: ADVICE | To avoid the slow RAM swap on 8GB GPUs, drop down to 'seedvr2_ema_3b-Q4_K_M.gguf' (2GB).")
                elif total_vram_gb <= 12.5:
                    log_node("SeedVR2 Upscale: ADVICE | For 12GB GPUs, 'seedvr2_ema_7b-Q4_K_M.gguf' (~4.8GB) is the sweet spot for 7B models.")
            
            if total_vram_gb <= 8.5:
                log_node("SeedVR2 Upscale: TIP | If SeedVR2 is still too slow due to VRAM limits, try using the 'Wireless UltimateSDUpscale' node instead. It is much lighter on memory!")
        else:
             log_node(f"SeedVR2 Upscale: VRAM Check OK | {total_vram_gb:.1f}GB total VRAM is plenty for model '{model}' (~{m_size_gb:.1f}GB).", color="GREEN")

        pil_image = tensor_to_pil(image)
        output_width = int(pil_image.width * upscale_by)
        output_height = int(pil_image.height * upscale_by)

        main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)

        output_image = process_and_stitch(
            tiles=main_tiles,
            width=output_width, height=output_height,
            dit_config=dit_config, vae_config=vae_config,
            seed=seed,
            tile_upscale_resolution=tile_upscale_resolution,
            upscale_factor=upscale_by,
            mask_blur=mask_blur,
            progress=None,
            original_image=pil_image,
            anti_aliasing_strength=anti_aliasing_strength,
            blending_method=blending_method,
            color_correction=color_correction,
        )

        # Cleanup VRAM after massive DiT model use
        import gc
        import comfy.model_management as mm
        log_node("SeedVR2 Upscale: Finished | VRAM cleared", color="GREEN")
        mm.soft_empty_cache()
        gc.collect()

        return (pil_to_tensor(output_image),)

# --- Face Detailers ---

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
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def load_bbox(self, model_name):
        try:
            bbox_detector = detector.load_bbox_model(model_name)
            return (bbox_detector,)
        except Exception as e:
            log_node(f"Error loading BBox Detector: {e}", color="RED")
            return (None,)

class UmeAiRT_WirelessFaceDetailer_Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "image": ("IMAGE",),
                 "bbox_detector": ("BBOX_DETECTOR",),
                 "enabled": ("BOOLEAN", {"default": True}),
                 "guide_size": ("INT", {"default": 512, "min": 64, "max": 2048}),
                 "max_size": ("INT", {"default": 1024, "min": 64, "max": 2048}),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def face_detail(self, image, bbox_detector, enabled, guide_size, max_size, denoise):
        if not enabled: return (image,)
        
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        clip = UME_SHARED_STATE.get(KEY_CLIP)
        
        steps = int(UME_SHARED_STATE.get(KEY_STEPS, 20))
        cfg = float(UME_SHARED_STATE.get(KEY_CFG, 8.0))
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        
        pos_text = str(UME_SHARED_STATE.get(KEY_POSITIVE, ""))
        neg_text = str(UME_SHARED_STATE.get(KEY_NEGATIVE, ""))

        if not model or not vae or not clip:
            return (image,)
            
        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(neg_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative = [[cond, {"pooled_output": pooled}]]

        segs = bbox_detector.detect(image, 0.5, 10, 3.0, 10)
        
        with SamplerContext():
             result = fd_logic.do_detail(
                 image=image, segs=segs, model=model, clip=clip, vae=vae,
                 guide_size=guide_size, guide_size_for_bbox=True, max_size=max_size,
                 seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                 positive=positive, negative=negative, denoise=denoise,
                 feather=5, noise_mask=True, force_inpaint=True, drop_size=10
             )
        return result

class UmeAiRT_WirelessFaceDetailer_Simple(UmeAiRT_WirelessFaceDetailer_Advanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "image": ("IMAGE",),
                 "bbox_detector": ("BBOX_DETECTOR",),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
        
    def face_detail(self, image, bbox_detector, denoise):
        return super().face_detail(image, bbox_detector, True, 512, 1024, denoise)

# --- Detailer Daemon ---

def make_detail_daemon_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    if mid_idx + 1 > start_idx:
        multipliers[start_idx : mid_idx + 1] = start_values
    
    if end_idx + 1 > mid_idx:
        multipliers[mid_idx : end_idx + 1] = end_values
        
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers

def get_dd_schedule(sigma, sigmas, dd_schedule):
    sched_len = len(dd_schedule)
    if sched_len < 2 or len(sigmas) < 2 or sigma <= 0 or not (sigmas[-1] <= sigma <= sigmas[0]):
        return 0.0
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (idx == 0 and sigma >= sigmas[0]) or (idx == sched_len - 1 and sigma <= sigmas[-2]) or deltas[idx] == 0:
        return dd_schedule[idx].item()
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0: return dd_schedule[idxlow]
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()

def detail_daemon_sampler(model, x, sigmas, *, dds_wrapped_sampler, dds_make_schedule, dds_cfg_scale_override, **kwargs):
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        
    dd_schedule = torch.tensor(dds_make_schedule(len(sigmas) - 1), dtype=torch.float32, device="cpu")
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x, sigma, **extra_args):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in ("inner_model", "sigmas"):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
            
    return dds_wrapped_sampler.sampler_function(
        model_wrapper, x, sigmas, **kwargs, **dds_wrapped_sampler.extra_options,
    )

class UmeAiRT_Detailer_Daemon_Simple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough"}),
                "detail_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def process(self, enabled, detail_amount, image=None):
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        start_image = image if image is not None else UME_SHARED_STATE.get(KEY_IMAGE)
        
        if not enabled:
            if start_image is None: return (torch.zeros((1, 512, 512, 3)),)
            return (start_image,)

        steps = UME_SHARED_STATE.get(KEY_STEPS, 20)
        cfg = UME_SHARED_STATE.get(KEY_CFG, 8.0)
        sampler_name = UME_SHARED_STATE.get(KEY_SAMPLER, "euler")
        scheduler = UME_SHARED_STATE.get(KEY_SCHEDULER, "normal")
        seed = int(UME_SHARED_STATE.get(KEY_SEED, 0))
        
        denoise = 0.5
        refine_denoise = 0.05
        positive = UME_SHARED_STATE.get(KEY_POSITIVE)
        negative = UME_SHARED_STATE.get(KEY_NEGATIVE)

        if any(x is None for x in [model, vae, start_image, positive, negative]):
            log_node("Missing Wireless Context for Detailer Daemon", color="RED")
            return (torch.zeros((1, 512, 512, 3)),)

        # Prompt Encoding check
        if isinstance(positive, str) or isinstance(negative, str):
            clip = UME_SHARED_STATE.get(KEY_CLIP)
            if not clip: return (start_image,)
            if isinstance(positive, str):
                tokens = clip.tokenize(positive)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                positive = [[cond, {"pooled_output": pooled}]]
            if isinstance(negative, str):
                tokens = clip.tokenize(negative)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                negative = [[cond, {"pooled_output": pooled}]]

        t = vae.encode(start_image[:,:,:,:3])
        latent_image = {"samples": t}

        def dds_make_schedule(num_steps):
            return make_detail_daemon_schedule(
                num_steps, start=0.2, end=0.8, bias=0.5, amount=detail_amount, exponent=1.0,
                start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True
            )
        
        sampler_obj = comfy.samplers.KSampler(
             model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options
        )
        base_low_level_sampler = comfy.samplers.sampler_object(sampler_name)

        class DD_Sampler_Wrapper:
            def __init__(self, base_sampler, make_sched, cfg_override):
                self.base_sampler = base_sampler
                self.make_sched = make_sched
                self.cfg = cfg_override

            def __call__(self, model, x, sigmas, *args, **kwargs):
                return detail_daemon_sampler(
                    model, x, sigmas,
                    dds_wrapped_sampler=self.base_sampler, dds_make_schedule=self.make_sched, dds_cfg_scale_override=self.cfg,
                    **kwargs
                )
        
        dd_wrapper_func = DD_Sampler_Wrapper(base_low_level_sampler, dds_make_schedule, cfg)
        wrapped_sampler = comfy.samplers.KSAMPLER(dd_wrapper_func, extra_options=base_low_level_sampler.extra_options, inpaint_options=base_low_level_sampler.inpaint_options)
        
        sigmas = sampler_obj.sigmas
        noise = torch.randn(latent_image["samples"].size(), dtype=latent_image["samples"].dtype, layout=latent_image["samples"].layout, generator=torch.manual_seed(seed), device="cpu")

        samples = comfy.sample.sample_custom(
            model, noise, cfg, wrapped_sampler, sigmas, positive, negative, latent_image["samples"], noise_mask=None, callback=None, disable_pbar=False, seed=seed
        )

        log_node(f"Detail Daemon: Processing | Amount: {detail_amount} | Steps: {steps} | Denoise: {denoise}")

        if refine_denoise > 0.0:
            refine_steps = max(1, int(steps * 0.25))
            refine_sampler_obj = comfy.samplers.KSampler(model, steps=refine_steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=refine_denoise, model_options=model.model_options)
            refine_sigmas = refine_sampler_obj.sigmas
            refine_noise = torch.randn(samples.size(), dtype=samples.dtype, layout=samples.layout, generator=torch.manual_seed(seed+1), device="cpu")
            samples = comfy.sample.sample_custom(
                 model, refine_noise, cfg, comfy.samplers.sampler_object(sampler_name), refine_sigmas, positive, negative, samples, noise_mask=None, callback=None, disable_pbar=False, seed=seed+1
            )
        
        decoded = vae.decode(samples)
        UME_SHARED_STATE[KEY_IMAGE] = decoded
        log_node("Detail Daemon: Finished", color="GREEN")
        return (decoded,)

class UmeAiRT_Detailer_Daemon_Advanced(UmeAiRT_Detailer_Daemon_Simple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detail_amount": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_denoise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 20}),
                "refine_steps": ("INT", {"default": 2}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "seed": ("INT", {"default": 0}),
            }
        }
    
    FUNCTION = "process_advanced"
    
    def process_advanced(self, detail_amount, start, end, bias, exponent, start_offset, end_offset, fade, smooth, denoise, refine_denoise, image=None, steps=20, refine_steps=2, cfg=8.0, sampler_name="euler", scheduler="normal", seed=0):
        # Implementation mirrors Simple but uses passed args
        model = UME_SHARED_STATE.get(KEY_MODEL)
        vae = UME_SHARED_STATE.get(KEY_VAE)
        start_image = image if image is not None else UME_SHARED_STATE.get(KEY_IMAGE)
        positive = UME_SHARED_STATE.get(KEY_POSITIVE)
        negative = UME_SHARED_STATE.get(KEY_NEGATIVE)
        
        if any(x is None for x in [model, vae, start_image, positive, negative]):
            return (torch.zeros((1, 512, 512, 3)),)

        if isinstance(positive, str) or isinstance(negative, str):
            clip = UME_SHARED_STATE.get(KEY_CLIP)
            if not clip: return (start_image,)
            if isinstance(positive, str):
                tokens = clip.tokenize(positive)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                positive = [[cond, {"pooled_output": pooled}]]
            if isinstance(negative, str):
                tokens = clip.tokenize(negative)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                negative = [[cond, {"pooled_output": pooled}]]

        t = vae.encode(start_image[:,:,:,:3])
        latent_image = {"samples": t}

        def dds_make_schedule(num_steps):
            return make_detail_daemon_schedule(
                num_steps, start=start, end=end, bias=bias, amount=detail_amount, exponent=exponent,
                start_offset=start_offset, end_offset=end_offset, fade=fade, smooth=smooth
            )

        sampler_obj = comfy.samplers.KSampler(
             model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options
        )
        base_low_level_sampler = comfy.samplers.sampler_object(sampler_name)

        class DD_Sampler_Wrapper:
            def __init__(self, base_sampler, make_sched, cfg_override):
                self.base_sampler = base_sampler
                self.make_sched = make_sched
                self.cfg = cfg_override
            def __call__(self, model, x, sigmas, *args, **kwargs):
                return detail_daemon_sampler(
                    model, x, sigmas,
                    dds_wrapped_sampler=self.base_sampler, dds_make_schedule=self.make_sched, dds_cfg_scale_override=self.cfg,
                    **kwargs
                )
        
        dd_wrapper_func = DD_Sampler_Wrapper(base_low_level_sampler, dds_make_schedule, cfg)
        wrapped_sampler = comfy.samplers.KSAMPLER(dd_wrapper_func, extra_options=base_low_level_sampler.extra_options, inpaint_options=base_low_level_sampler.inpaint_options)
        
        sigmas = sampler_obj.sigmas
        noise = torch.randn(latent_image["samples"].size(), dtype=latent_image["samples"].dtype, layout=latent_image["samples"].layout, generator=torch.manual_seed(seed), device="cpu")

        samples = comfy.sample.sample_custom(
            model, noise, cfg, wrapped_sampler, sigmas, positive, negative, latent_image["samples"], noise_mask=None, callback=None, disable_pbar=False, seed=seed
        )

        log_node(f"Detail Daemon: Processing | Amount: {detail_amount} | Steps: {steps} | Denoise: {denoise}")

        if refine_denoise > 0.0:
            refine_sampler_obj = comfy.samplers.KSampler(model, steps=refine_steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=refine_denoise, model_options=model.model_options)
            refine_sigmas = refine_sampler_obj.sigmas
            refine_noise = torch.randn(samples.size(), dtype=samples.dtype, layout=samples.layout, generator=torch.manual_seed(seed+1), device="cpu")
            samples = comfy.sample.sample_custom(
                 model, refine_noise, cfg, comfy.samplers.sampler_object(sampler_name), refine_sigmas, positive, negative, samples, noise_mask=None, callback=None, disable_pbar=False, seed=seed+1
            )
        
        decoded = vae.decode(samples)
        UME_SHARED_STATE[KEY_IMAGE] = decoded
        log_node("Detail Daemon: Finished", color="GREEN")
        return (decoded,)

