"""
UmeAiRT Toolkit - Logic Nodes
------------------------------
Processor nodes migrated to use GenerationContext pipeline instead of global state.
"""

import torch
import numpy as np
import os
import folder_paths
import comfy.utils
import comfy.sd
import nodes as comfy_nodes
import nodes as comfy_nodes
from .common import log_node, encode_prompts
from .logger import logger

# Try import internals
try:
    from .seedvr2_adapter import execute_seedvr2
    from .stitching import process_and_stitch
except ImportError as e:
    from .logger import log_node
    log_node(f"Logic Nodes: Could not import SeedVR2 internals: {e}", color="YELLOW")
    pass

try:
    from .facedetailer_core import logic as fd_logic
    from .facedetailer_core import detector
except ImportError as e:
    from .logger import log_node
    log_node(f"Logic Nodes: Could not import FaceDetailer internals: {e}", color="YELLOW")
    pass


# --- Helpers ---

from .optimization_utils import SamplerContext


# --- VRAM Management ---

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


# --- Base Classes (used by block_nodes.py) ---

class UmeAiRT_UltimateUpscale_Base:
    """Base class providing prompt encoding utilities for Ultimate Upscaler nodes."""
    def encode_prompts(self, clip, pos_text, neg_text):
        return encode_prompts(clip, pos_text, neg_text)

    def get_usdu_node(self):
        import sys
        import os
        usdu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usdu_core")
        if usdu_path not in sys.path:
            sys.path.append(usdu_path)
        try:
            import usdu_main
            return usdu_main.UltimateSDUpscale()
        except ImportError as e:
            raise ImportError(f"UmeAiRT: Could not load bundled UltimateSDUpscale node from usdu_core. Error: {e}")


# --- Standalone Utility Nodes ---

class UmeAiRT_BboxDetectorLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("bbox"), {"tooltip": "Choose the face detection model (.pt file from models/bbox/ folder)."}),
            }
        }
    RETURN_TYPES = ("BBOX_DETECTOR",)
    FUNCTION = "load_bbox"
    CATEGORY = "UmeAiRT/Loaders"

    def load_bbox(self, model_name):
        try:
            bbox_detector = detector.load_bbox_model(model_name)
            return (bbox_detector,)
        except Exception as e:
            log_node(f"Error loading BBox Detector: {e}", color="RED")
            raise RuntimeError(f"BBox Detector: Failed to load '{model_name}': {e}")


# --- Pipeline-Aware UltimateUpscale ---

class UmeAiRT_PipelineUltimateUpscale(UmeAiRT_UltimateUpscale_Base):
    """Simple Ultimate SD Upscale — reads image and models from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
            },
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def upscale(self, gen_pipe, enabled, model, upscale_by):
        image = gen_pipe.image
        if image is None:
            raise ValueError("UltimateUpscale: No image in pipeline.")
        log_node(f"UltimateSDUpscale (Simple): Processing | Ratio: x{upscale_by} | Model: {model}")
        if not enabled:
            return (gen_pipe,)

        denoise = 0.35
        clean_prompt = True
        mode_type = "Linear"
        tile_padding = 32

        sd_model = gen_pipe.model
        vae = gen_pipe.vae
        clip = gen_pipe.clip

        steps = max(5, int(gen_pipe.steps or 20) // 4)
        cfg = 1.0
        sampler_name = gen_pipe.sampler_name or "euler"
        scheduler = gen_pipe.scheduler or "normal"
        seed = int(gen_pipe.seed or 0)

        if not sd_model or not vae or not clip:
            raise ValueError("UltimateUpscale: Missing Model/VAE/CLIP in pipeline.")

        pos_text = str(gen_pipe.positive_prompt or "")
        neg_text = str(gen_pipe.negative_prompt or "")
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)

        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        usdu_node = self.get_usdu_node()

        tile_width = int(gen_pipe.width or 1024)
        tile_height = int(gen_pipe.height or 1024)

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

        ctx = gen_pipe.clone()
        ctx.image = res[0]
        return (ctx,)


class UmeAiRT_PipelineUltimateUpscale_Advanced(UmeAiRT_UltimateUpscale_Base):
    """Advanced Ultimate SD Upscale — reads models/settings from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        seam_fix_modes = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
        return {
             "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
            },
            "optional":{
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Pipeline Prompt", "tooltip": "Simplify the prompt during upscaling to prevent the AI from adding unwanted new elements."}),
                "mode_type": (usdu_modes, {"default": "Linear", "tooltip": "How tiles are arranged: Linear (rows), Chess (checkerboard for fewer seams), or None."}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "Width of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "Height of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "Softens tile edges for smoother blending between tiles."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128, "tooltip": "How much tiles overlap in pixels. More overlap = smoother transitions, but slower."}),
                "seam_fix_mode": (seam_fix_modes, {"default": "None", "tooltip": "Extra pass to remove visible lines between tiles. 'Band+Half' recommended."}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much to redraw along tile seams. 0.3-0.5 usually works well."}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 512, "tooltip": "Width of the band processed along each seam in pixels."}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "tooltip": "Softens the seam fix mask for gradual blending."}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "tooltip": "Additional context padding around each seam fix area."}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True, "tooltip": "Ensure all tiles are identical size. May crop slightly but prevents edge artifacts."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "tooltip": "Decode the image in tiles instead of all at once. Saves VRAM on large images."}),
            }
        }
    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def upscale(self, gen_pipe, model, upscale_by, denoise, clean_prompt=True, mode_type="Linear",
                tile_width=512, tile_height=512, mask_blur=8, tile_padding=32,
                seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_width=64,
                seam_fix_mask_blur=8, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False):

        image = gen_pipe.image
        if image is None:
            raise ValueError("UltimateUpscale Advanced: No image in pipeline.")
        log_node(f"UltimateSDUpscale (Advanced): Processing | Ratio: x{upscale_by} | Model: {model} | Denoise: {denoise}")

        sd_model = gen_pipe.model
        vae = gen_pipe.vae
        clip = gen_pipe.clip

        steps = max(5, int(gen_pipe.steps or 20) // 4)
        cfg = 1.0
        sampler_name = gen_pipe.sampler_name or "euler"
        scheduler = gen_pipe.scheduler or "normal"
        seed = int(gen_pipe.seed or 0)

        if not sd_model or not vae or not clip:
            raise ValueError("UltimateUpscale Advanced: Missing Model/VAE/CLIP in pipeline.")

        pos_text = str(gen_pipe.positive_prompt or "")
        neg_text = str(gen_pipe.negative_prompt or "")
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)

        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        usdu_node = self.get_usdu_node()

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

        ctx = gen_pipe.clone()
        ctx.image = res[0]
        return (ctx,)


# --- Pipeline-Aware SeedVR2 Upscale ---

class UmeAiRT_PipelineSeedVR2Upscale:
    """SeedVR2 upscaler — reads seed from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        KNOWN_DIT_MODELS = [
            "seedvr2_ema_3b-Q4_K_M.gguf", "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_fp16.safetensors", "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
        ]
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"

        try:
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception as e:
             from .logger import log_node
             log_node(f"SeedVR2 Upscale: Could not map extended models: {e}", color="YELLOW")
             dit_models = KNOWN_DIT_MODELS

        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "model": (dit_models, {"default": default_dit, "tooltip": "Choose the SeedVR2 upscale model."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
            },
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    @staticmethod
    def _build_configs(model_name: str):
        """Build dit_config and vae_config dicts."""
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dit_config = {
            "model": model_name, "device": device, "offload_device": "cpu",
            "cache_model": False, "blocks_to_swap": 0, "swap_io_components": False,
            "attention_mode": "sdpa", "torch_compile_args": None, "node_id": None,
        }
        vae_config = {
            "model": "ema_vae_fp16.safetensors", "device": device, "offload_device": "cpu",
            "cache_model": False, "encode_tiled": False, "encode_tile_size": 1024,
            "encode_tile_overlap": 128, "decode_tiled": False, "decode_tile_size": 1024,
            "decode_tile_overlap": 128, "tile_debug": "false", "torch_compile_args": None, "node_id": None,
        }
        return dit_config, vae_config

    def upscale(self, gen_pipe, enabled, model, upscale_by):
        if not enabled:
            return (gen_pipe,)

        image = gen_pipe.image
        if image is None:
            raise ValueError("SeedVR2 Upscale: No image in pipeline.")

        try:
            from ..seedvr2_core.image_utils import tensor_to_pil, pil_to_tensor
            from ..seedvr2_core.tiling import generate_tiles
            from ..seedvr2_core.stitching import process_and_stitch
        except ImportError:
             raise ImportError("SeedVR2 Core modules not found in '../seedvr2_core'. Verify installation.")

        seed = int(gen_pipe.seed or 100) % (2**32)
        dit_config, vae_config = self._build_configs(model)

        log_node(f"SeedVR2 Upscale: Processing | Ratio: x{upscale_by} | Model: {model} | Seed: {seed}")
        _ensure_vram_for_seedvr2()

        import comfy.model_management as mm
        device = mm.get_torch_device()
        total_vram_gb = mm.get_total_memory(device) / (1024**3)

        model_l = model.lower()
        if "7b" in model_l:
            if "q4" in model_l: m_size_gb = 4.8
            elif "fp16" in model_l and "mixed" not in model_l: m_size_gb = 16.5
            else: m_size_gb = 8.5
        else:
            if "q4" in model_l: m_size_gb = 2.0
            elif "q8" in model_l: m_size_gb = 3.7
            elif "fp16" in model_l: m_size_gb = 6.8
            else: m_size_gb = 3.4

        overhead_gb = 3.5
        req_vram = m_size_gb + overhead_gb

        if total_vram_gb < req_vram:
            if total_vram_gb < m_size_gb:
                log_node(f"SeedVR2 Upscale: CRITICAL | Model '{model}' (~{m_size_gb:.1f}GB) > total VRAM ({total_vram_gb:.1f}GB)!", color="RED")
            else:
                log_node(f"SeedVR2 Upscale: WARNING | VRAM tight for '{model}' (~{m_size_gb:.1f}GB).", color="ORANGE")
        else:
             log_node(f"SeedVR2 Upscale: VRAM Check OK | {total_vram_gb:.1f}GB total.", color="GREEN")

        tile_width, tile_height = 512, 512
        mask_blur, tile_padding = 0, 32
        tile_upscale_resolution = 1024
        tiling_strategy = "Chess"
        anti_aliasing_strength = 0.0
        blending_method = "auto"
        color_correction = "lab"

        pil_image = tensor_to_pil(image)
        output_width = int(pil_image.width * upscale_by)
        output_height = int(pil_image.height * upscale_by)

        main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)

        output_image = process_and_stitch(
            tiles=main_tiles, width=output_width, height=output_height,
            dit_config=dit_config, vae_config=vae_config, seed=seed,
            tile_upscale_resolution=tile_upscale_resolution, upscale_factor=upscale_by,
            mask_blur=mask_blur, progress=None, original_image=pil_image,
            anti_aliasing_strength=anti_aliasing_strength,
            blending_method=blending_method, color_correction=color_correction,
        )

        import gc
        import comfy.model_management as mm
        log_node("SeedVR2 Upscale: Finished | VRAM cleared", color="GREEN")
        mm.soft_empty_cache()
        gc.collect()

        ctx = gen_pipe.clone()
        ctx.image = pil_to_tensor(output_image)
        return (ctx,)


class UmeAiRT_PipelineSeedVR2Upscale_Advanced:
    """Advanced SeedVR2 upscaler with full control — reads seed from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        KNOWN_DIT_MODELS = [
            "seedvr2_ema_3b-Q4_K_M.gguf", "seedvr2_ema_3b-Q8_0.gguf",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_7b-Q4_K_M.gguf", "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
            "seedvr2_ema_7b_fp16.safetensors", "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors", "seedvr2_ema_7b_sharp_fp16.safetensors",
        ]
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        try:
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception as e:
             from .logger import log_node
             log_node(f"SeedVR2 Advanced: Could not map extended models: {e}", color="YELLOW")
             dit_models = KNOWN_DIT_MODELS

        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "model": (dit_models, {"default": default_dit, "tooltip": "Choose the upscale model (e.g. 4x-UltraSharp, RealESRGAN)."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Width of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "tooltip": "Height of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Softens tile edges for smoother blending between tiles."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8, "tooltip": "How much tiles overlap in pixels. More overlap = smoother transitions, but slower."}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "tooltip": "Resolution for each tile during upscaling processing."}),
                "tiling_strategy": (["Chess", "Linear"], {"default": "Chess", "tooltip": "Controls how tiles are positioned and processed during upscaling."}),
                "anti_aliasing_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Reduces jagged edges between tiles. Higher = smoother but may reduce sharpness."}),
                "blending_method": (["auto", "multiband", "bilateral", "content_aware", "linear", "simple"], {"default": "auto", "tooltip": "Algorithm for merging overlapping tile regions (e.g. linear, gaussian)."}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab", "tooltip": "Match colors between adjacent tiles to prevent visible color shifts."}),
            },
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def upscale(self, gen_pipe, enabled, model, upscale_by,
                tile_width, tile_height, mask_blur, tile_padding,
                tile_upscale_resolution, tiling_strategy,
                anti_aliasing_strength, blending_method, color_correction):
        if not enabled:
            return (gen_pipe,)

        image = gen_pipe.image
        if image is None:
            raise ValueError("SeedVR2 Advanced: No image in pipeline.")

        try:
            from ..seedvr2_core.image_utils import tensor_to_pil, pil_to_tensor
            from ..seedvr2_core.tiling import generate_tiles
            from ..seedvr2_core.stitching import process_and_stitch
        except ImportError:
             raise ImportError("SeedVR2 Core modules not found. Verify installation.")

        seed = int(gen_pipe.seed or 100) % (2**32)
        dit_config, vae_config = UmeAiRT_PipelineSeedVR2Upscale._build_configs(model)

        log_node(f"SeedVR2 Upscale: Processing | Ratio: x{upscale_by} | Model: {model} | Seed: {seed}")
        _ensure_vram_for_seedvr2()

        pil_image = tensor_to_pil(image)
        output_width = int(pil_image.width * upscale_by)
        output_height = int(pil_image.height * upscale_by)

        main_tiles = generate_tiles(pil_image, tile_width, tile_height, tile_padding, tiling_strategy)

        output_image = process_and_stitch(
            tiles=main_tiles, width=output_width, height=output_height,
            dit_config=dit_config, vae_config=vae_config, seed=seed,
            tile_upscale_resolution=tile_upscale_resolution, upscale_factor=upscale_by,
            mask_blur=mask_blur, progress=None, original_image=pil_image,
            anti_aliasing_strength=anti_aliasing_strength,
            blending_method=blending_method, color_correction=color_correction,
        )

        import gc
        import comfy.model_management as mm
        log_node("SeedVR2 Upscale: Finished | VRAM cleared", color="GREEN")
        mm.soft_empty_cache()
        gc.collect()

        ctx = gen_pipe.clone()
        ctx.image = pil_to_tensor(output_image)
        return (ctx,)


# --- Pipeline-Aware Face Detailers ---

class UmeAiRT_PipelineFaceDetailer_Advanced:
    """Face detailer — reads image and models from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                 "bbox_detector": ("BBOX_DETECTOR",),
                 "enabled": ("BOOLEAN", {"default": True, "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                 "guide_size": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": "Target face crop size in pixels. Larger = more detail but slower. 384-512 recommended."}),
                 "max_size": ("INT", {"default": 1024, "min": 64, "max": 2048, "tooltip": "Maximum allowed face crop. Prevents excessive VRAM usage on very large faces."}),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def face_detail(self, gen_pipe, bbox_detector, enabled, guide_size, max_size, denoise):
        image = gen_pipe.image
        if image is None:
            raise ValueError("FaceDetailer: No image in pipeline.")
        if not enabled: return (gen_pipe,)

        model = gen_pipe.model
        vae = gen_pipe.vae
        clip = gen_pipe.clip

        steps = int(gen_pipe.steps or 20)
        cfg = float(gen_pipe.cfg or 8.0)
        sampler_name = gen_pipe.sampler_name or "euler"
        scheduler = gen_pipe.scheduler or "normal"
        seed = int(gen_pipe.seed or 0)

        pos_text = str(gen_pipe.positive_prompt or "")
        neg_text = str(gen_pipe.negative_prompt or "")

        if not model or not vae or not clip:
            return (gen_pipe,)

        positive, negative = encode_prompts(clip, pos_text, neg_text)

        segs = bbox_detector.detect(image, 0.5, 10, 3.0, 10)

        result = fd_logic.do_detail(
                 image=image, segs=segs, model=model, clip=clip, vae=vae,
                 guide_size=guide_size, guide_size_for_bbox=True, max_size=max_size,
                 seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                 positive=positive, negative=negative, denoise=denoise,
                 feather=5, noise_mask=True, force_inpaint=True, drop_size=10
             )
        ctx = gen_pipe.clone()
        ctx.image = result[0]
        return (ctx,)


class UmeAiRT_PipelineFaceDetailer(UmeAiRT_PipelineFaceDetailer_Advanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                 "bbox_detector": ("BBOX_DETECTOR",),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
            }
        }

    def face_detail(self, gen_pipe, bbox_detector, denoise):
        return super().face_detail(gen_pipe, bbox_detector, True, 512, 1024, denoise)


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
    """Detail daemon — reads models/settings from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "detail_amount": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": "How much extra detail to add. Positive = sharpen, negative = soften. Start with 0.1-0.5."}),
            },
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def process(self, gen_pipe, enabled, detail_amount):
        if not enabled:
            return (gen_pipe,)

        start_image = gen_pipe.image
        if start_image is None:
            raise ValueError("Detail Daemon: No image in pipeline.")

        model = gen_pipe.model
        vae = gen_pipe.vae

        steps = int(gen_pipe.steps or 20)
        cfg = float(gen_pipe.cfg or 8.0)
        sampler_name = gen_pipe.sampler_name or "euler"
        scheduler = gen_pipe.scheduler or "normal"
        seed = int(gen_pipe.seed or 0)

        denoise = 0.5
        refine_denoise = 0.05
        clip = gen_pipe.clip
        pos_text = str(gen_pipe.positive_prompt or "")
        neg_text = str(gen_pipe.negative_prompt or "")

        if any(x is None for x in [model, vae, start_image, clip]):
            log_node("Missing Pipeline Context for Detailer Daemon", color="RED")
            return (gen_pipe,)

        # Encode prompts
        positive, negative = encode_prompts(clip, pos_text, neg_text)

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

        log_node(f"Detail Daemon: Processing | Amount: {detail_amount} | Steps: {steps} | Denoise: {denoise}")

        samples = comfy.sample.sample_custom(
            model, noise, cfg, wrapped_sampler, sigmas, positive, negative, latent_image["samples"], noise_mask=None, callback=None, disable_pbar=False, seed=seed
        )

        if refine_denoise > 0.0:
            refine_sampler_obj = comfy.samplers.KSampler(model, steps=2, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=refine_denoise, model_options=model.model_options)
            refine_sigmas = refine_sampler_obj.sigmas
            refine_noise = torch.randn(samples.size(), dtype=samples.dtype, layout=samples.layout, generator=torch.manual_seed(seed+1), device="cpu")
            samples = comfy.sample.sample_custom(
                 model, refine_noise, cfg, comfy.samplers.sampler_object(sampler_name), refine_sigmas, positive, negative, samples, noise_mask=None, callback=None, disable_pbar=False, seed=seed+1
            )

        decoded = vae.decode(samples)
        log_node("Detail Daemon: Finished", color="GREEN")
        ctx = gen_pipe.clone()
        ctx.image = decoded
        return (ctx,)


class UmeAiRT_Detailer_Daemon_Advanced(UmeAiRT_Detailer_Daemon_Simple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "detail_amount": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01, "tooltip": "How much extra detail to add. Positive = sharpen, negative = soften. Start with 0.1-0.5."}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "When detail enhancement begins in the sampling process (0.0 = from the start)."}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "When detail enhancement ends in the sampling process (1.0 = until completion)."}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Shifts the peak of detail enhancement within the start-end range."}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Controls how sharply the detail effect ramps up/down. Higher = more concentrated."}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Fine-tune the exact starting point of the detail curve."}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Fine-tune the exact ending point of the detail curve."}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Gradually reduce the effect at the edges of the schedule range."}),
                "smooth": ("BOOLEAN", {"default": True, "tooltip": "Smooth out the detail schedule curve to avoid abrupt changes."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
                "refine_denoise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much to redraw in the refinement pass. Lower = preserves more original detail."}),
            },
            "optional": {
                "steps": ("INT", {"default": 20, "tooltip": "Total sampling steps. More steps = better quality but slower. 20-30 recommended."}),
                "refine_steps": ("INT", {"default": 2, "tooltip": "Steps for the extra refinement pass. 5-10 usually sufficient."}),
                "cfg": ("FLOAT", {"default": 8.0, "tooltip": "How closely the AI follows your prompt. Higher = more literal but may look artificial. 5-8 recommended."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "seed": ("INT", {"default": 0, "tooltip": "Seed for reproducible results. Same seed + same settings = same image."}),
            }
        }

    FUNCTION = "process_advanced"

    def process_advanced(self, gen_pipe, detail_amount, start, end, bias, exponent, start_offset, end_offset, fade, smooth, denoise, refine_denoise, steps=20, refine_steps=2, cfg=8.0, sampler_name="euler", scheduler="normal", seed=0):
        start_image = gen_pipe.image
        if start_image is None:
            raise ValueError("Detail Daemon Advanced: No image in pipeline.")

        model = gen_pipe.model
        vae = gen_pipe.vae
        clip = gen_pipe.clip
        pos_text = str(gen_pipe.positive_prompt or "")
        neg_text = str(gen_pipe.negative_prompt or "")

        if any(x is None for x in [model, vae, clip]):
            return (gen_pipe,)

        tokens = clip.tokenize(pos_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(neg_text)
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

        log_node(f"Detail Daemon: Processing | Amount: {detail_amount} | Steps: {steps} | Denoise: {denoise}")

        samples = comfy.sample.sample_custom(
            model, noise, cfg, wrapped_sampler, sigmas, positive, negative, latent_image["samples"], noise_mask=None, callback=None, disable_pbar=False, seed=seed
        )

        if refine_denoise > 0.0:
            refine_sampler_obj = comfy.samplers.KSampler(model, steps=refine_steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=refine_denoise, model_options=model.model_options)
            refine_sigmas = refine_sampler_obj.sigmas
            refine_noise = torch.randn(samples.size(), dtype=samples.dtype, layout=samples.layout, generator=torch.manual_seed(seed+1), device="cpu")
            samples = comfy.sample.sample_custom(
                 model, refine_noise, cfg, comfy.samplers.sampler_object(sampler_name), refine_sigmas, positive, negative, samples, noise_mask=None, callback=None, disable_pbar=False, seed=seed+1
            )

        decoded = vae.decode(samples)
        log_node("Detail Daemon Advanced: Finished", color="GREEN")
        ctx = gen_pipe.clone()
        ctx.image = decoded
        return (ctx,)
