"""
UmeAiRT Toolkit - SeedVR2 Upscale Nodes
-----------------------------------------
Pipeline-aware SeedVR2 upscaler nodes (Simple & Advanced).
"""

import gc
import comfy.model_management as mm
from .common import log_node, KNOWN_DIT_MODELS
from .logger import logger


# --- VRAM Management ---

SEEDVR2_VRAM_REQUIRED = 6 * 1024 * 1024 * 1024  # 6 GB
DECODE_VRAM_REQUIRED = 1.5 * 1024 * 1024 * 1024 # 1.5 GB

def _ensure_vram_for_decode():
    """Ensure sufficient VRAM for VAE Decode."""
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


# --- Lazy SeedVR2 Imports ---

_seedvr2_imports = {}

def _get_seedvr2_modules():
    """Lazy-load SeedVR2 core modules (import only once, fail loudly if missing)."""
    if not _seedvr2_imports:
        try:
            from ..seedvr2_core.image_utils import tensor_to_pil, pil_to_tensor
            from ..seedvr2_core.tiling import generate_tiles
            from ..seedvr2_core.stitching import process_and_stitch
            _seedvr2_imports.update({
                "tensor_to_pil": tensor_to_pil,
                "pil_to_tensor": pil_to_tensor,
                "generate_tiles": generate_tiles,
                "process_and_stitch": process_and_stitch,
            })
        except ImportError:
            raise ImportError("SeedVR2 Core modules not found in '../seedvr2_core'. Verify installation.")
    return _seedvr2_imports


# --- Pipeline-Aware SeedVR2 Upscale ---

class UmeAiRT_PipelineSeedVR2Upscale:
    """SeedVR2 upscaler — reads seed from pipeline.

    Tile/stitching parameters are hidden behind 'Show advanced inputs'.
    """
    @classmethod
    def INPUT_TYPES(s):
        default_dit = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"

        try:
             from ..seedvr2_core.seedvr2_adapter import _ensure_seedvr2_path
             _ensure_seedvr2_path()
             from seedvr2_videoupscaler.src.utils.constants import get_all_model_files
             on_disk = list(get_all_model_files().keys())
             extra = [f for f in on_disk if f not in KNOWN_DIT_MODELS and f != "ema_vae_fp16.safetensors"]
             dit_models = KNOWN_DIT_MODELS + sorted(extra)
        except Exception as e:
             log_node(f"SeedVR2 Upscale: Could not map extended models: {e}", color="YELLOW")
             dit_models = KNOWN_DIT_MODELS

        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "model": (dit_models, {"default": default_dit, "tooltip": "Choose the SeedVR2 upscale model."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "advanced": True, "tooltip": "Width of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "advanced": True, "tooltip": "Height of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "advanced": True, "tooltip": "Softens tile edges for smoother blending between tiles."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 8192, "step": 8, "advanced": True, "tooltip": "How much tiles overlap in pixels. More overlap = smoother transitions, but slower."}),
                "tile_upscale_resolution": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "advanced": True, "tooltip": "Resolution for each tile during upscaling processing."}),
                "tiling_strategy": (["Chess", "Linear"], {"default": "Chess", "advanced": True, "tooltip": "Controls how tiles are positioned and processed during upscaling."}),
                "anti_aliasing_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Reduces jagged edges between tiles. Higher = smoother but may reduce sharpness."}),
                "blending_method": (["auto", "multiband", "bilateral", "content_aware", "linear", "simple"], {"default": "auto", "advanced": True, "tooltip": "Algorithm for merging overlapping tile regions (e.g. linear, gaussian)."}),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {"default": "lab", "advanced": True, "tooltip": "Match colors between adjacent tiles to prevent visible color shifts."}),
            },
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Post-Process"

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

    def upscale(self, gen_pipe, enabled, model, upscale_by,
                tile_width=512, tile_height=512, mask_blur=0, tile_padding=32,
                tile_upscale_resolution=1024, tiling_strategy="Chess",
                anti_aliasing_strength=0.0, blending_method="auto", color_correction="lab"):
        if not enabled:
            return (gen_pipe,)

        image = gen_pipe.image
        if image is None:
            raise ValueError("SeedVR2 Upscale: No image in pipeline.")

        sv2 = _get_seedvr2_modules()

        seed = int(gen_pipe.seed or 100) % (2**32)
        dit_config, vae_config = self._build_configs(model)

        log_node(f"SeedVR2 Upscale: Processing | Ratio: x{upscale_by} | Model: {model} | Seed: {seed}")
        _ensure_vram_for_seedvr2()

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

        pil_image = sv2["tensor_to_pil"](image)
        output_width = int(pil_image.width * upscale_by)
        output_height = int(pil_image.height * upscale_by)

        main_tiles = sv2["generate_tiles"](pil_image, tile_width, tile_height, tile_padding, tiling_strategy)

        output_image = sv2["process_and_stitch"](
            tiles=main_tiles, width=output_width, height=output_height,
            dit_config=dit_config, vae_config=vae_config, seed=seed,
            tile_upscale_resolution=tile_upscale_resolution, upscale_factor=upscale_by,
            mask_blur=mask_blur, progress=None, original_image=pil_image,
            anti_aliasing_strength=anti_aliasing_strength,
            blending_method=blending_method, color_correction=color_correction,
        )

        log_node("SeedVR2 Upscale: Finished | VRAM cleared", color="GREEN")
        mm.soft_empty_cache()
        gc.collect()

        ctx = gen_pipe.clone()
        ctx.image = sv2["pil_to_tensor"](output_image)
        return (ctx,)
