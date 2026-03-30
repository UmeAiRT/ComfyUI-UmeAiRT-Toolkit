"""
UmeAiRT Toolkit - Ultimate SD Upscale Nodes
---------------------------------------------
Pipeline-aware UltimateSD Upscale nodes (Simple & Advanced).
"""

import os
import sys
import folder_paths
import comfy.samplers
from .common import log_node, encode_prompts, extract_pipeline_params


# --- Base Class ---

class UmeAiRT_UltimateUpscale_Base:
    """Base class providing prompt encoding utilities for Ultimate Upscaler nodes."""
    def encode_prompts(self, clip, pos_text, neg_text):
        return encode_prompts(clip, pos_text, neg_text)

    def get_usdu_node(self):
        usdu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usdu_core")
        added_to_path = False
        if usdu_path not in sys.path:
            sys.path.insert(0, usdu_path)
            added_to_path = True
        try:
            import usdu_main
            return usdu_main.UltimateSDUpscale()
        except ImportError as e:
            raise ImportError(f"UmeAiRT: Could not load bundled UltimateSDUpscale node from usdu_core. Error: {e}")
        finally:
            if added_to_path and usdu_path in sys.path:
                sys.path.remove(usdu_path)


# --- Pipeline-Aware UltimateUpscale ---

class UmeAiRT_PipelineUltimateUpscale(UmeAiRT_UltimateUpscale_Base):
    """Ultimate SD Upscale — reads models/settings from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        usdu_modes = ["Linear", "Chess", "None"]
        seam_fix_modes = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
        samplers = ["Pipeline"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["Pipeline"] + comfy.samplers.KSampler.SCHEDULERS
        return {
             "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider", "tooltip": "How much to enlarge the image (e.g. 2.0 = double the resolution)."}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "advanced": True, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
            },
            "optional":{
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Reduces Hallucinations", "label_off": "Use Pipeline Prompt", "advanced": True, "tooltip": "Simplify the prompt during upscaling to prevent the AI from adding unwanted new elements."}),
                "upscale_steps": ("INT", {"default": 0, "min": 0, "max": 150, "advanced": True, "tooltip": "Tile steps. Leave at 0 for auto calculation (Pipeline Steps / 4)."}),
                "upscale_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.5, "advanced": True, "tooltip": "Tile CFG Guidance. Default 1.0 (recommended), increase for stricter prompt adherence."}),
                "upscale_sampler": (samplers, {"default": "Pipeline", "advanced": True, "tooltip": "Force a specific sampler for upscaling, or use the pipeline's."}),
                "upscale_scheduler": (schedulers, {"default": "Pipeline", "advanced": True, "tooltip": "Force a specific noise scheduler for upscaling, or use the pipeline's."}),
                "mode_type": (usdu_modes, {"default": "Linear", "advanced": True, "tooltip": "How tiles are arranged: Linear (rows), Chess (checkerboard for fewer seams), or None."}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 2048, "advanced": True, "tooltip": "Width of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 2048, "advanced": True, "tooltip": "Height of each tile in pixels. Smaller = less VRAM but slower. 512-1024 recommended."}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "advanced": True, "tooltip": "Softens tile edges for smoother blending between tiles."}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 128, "advanced": True, "tooltip": "How much tiles overlap in pixels. More overlap = smoother transitions, but slower."}),
                "seam_fix_mode": (seam_fix_modes, {"default": "None", "advanced": True, "tooltip": "Extra pass to remove visible lines between tiles. 'Band+Half' recommended."}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "How much to redraw along tile seams. 0.3-0.5 usually works well."}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 512, "advanced": True, "tooltip": "Width of the band processed along each seam in pixels."}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "advanced": True, "tooltip": "Softens the seam fix mask for gradual blending."}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "advanced": True, "tooltip": "Additional context padding around each seam fix area."}),
                "force_uniform_tiles": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Ensure all tiles are identical size. May crop slightly but prevents edge artifacts."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "advanced": True, "tooltip": "Decode the image in tiles instead of all at once. Saves VRAM on large images."}),
            }
        }
    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "upscale"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def upscale(self, gen_pipe, model, upscale_by, denoise=0.35, clean_prompt=True,
                upscale_steps=0, upscale_cfg=1.0, upscale_sampler="Pipeline", upscale_scheduler="Pipeline",
                mode_type="Linear", tile_width=512, tile_height=512, mask_blur=8, tile_padding=32,
                seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_width=64,
                seam_fix_mask_blur=8, seam_fix_padding=16, force_uniform_tiles=True, tiled_decode=False):

        image = gen_pipe.image
        if image is None:
            raise ValueError("UltimateUpscale: No image in pipeline.")
        log_node(f"UltimateSDUpscale: Processing | Ratio: x{upscale_by} | Model: {model} | Denoise: {denoise}")

        pp = extract_pipeline_params(gen_pipe)
        steps = max(5, pp.steps // 4) if upscale_steps == 0 else upscale_steps
        cfg = upscale_cfg
        
        final_sampler = pp.sampler_name if upscale_sampler == "Pipeline" else upscale_sampler
        final_scheduler = pp.scheduler if upscale_scheduler == "Pipeline" else upscale_scheduler

        target_pos_text = "" if clean_prompt else pp.pos_text
        positive, negative = self.encode_prompts(pp.clip, target_pos_text, pp.neg_text)

        try:
              from comfy_extras.nodes_upscale_model import UpscaleModelLoader
              upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
            raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")

        usdu_node = self.get_usdu_node()

        res = usdu_node.upscale(
                 image=image, model=pp.model, positive=positive, negative=negative, vae=pp.vae,
                 upscale_by=upscale_by, seed=pp.seed, steps=steps, cfg=cfg,
                 sampler_name=final_sampler, scheduler=final_scheduler, denoise=denoise,
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
