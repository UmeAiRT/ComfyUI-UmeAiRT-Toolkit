"""
UmeAiRT Toolkit - Face Detailer Nodes
---------------------------------------
Pipeline-aware FaceDetailer nodes and BBOX Detector loader.
"""

import folder_paths
from .common import log_node, encode_prompts, extract_pipeline_params

try:
    from .facedetailer_core import logic as fd_logic
    from .facedetailer_core import detector
except ImportError as e:
    log_node(f"Face Nodes: Could not import FaceDetailer internals: {e}", color="YELLOW")


# --- BBox Detector Loader ---

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
    CATEGORY = "UmeAiRT/Block/Loaders"

    def load_bbox(self, model_name):
        try:
            bbox_detector = detector.load_bbox_model(model_name)
            return (bbox_detector,)
        except Exception as e:
            log_node(f"Error loading BBox Detector: {e}", color="RED")
            raise RuntimeError(f"BBox Detector: Failed to load '{model_name}': {e}")


# --- Pipeline-Aware Face Detailers ---

class UmeAiRT_PipelineFaceDetailer:
    """Face detailer — reads image and models from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                 "bbox_detector": ("BBOX_DETECTOR",),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
                 "enabled": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                 "guide_size": ("INT", {"default": 512, "min": 64, "max": 2048, "advanced": True, "tooltip": "Target face crop size in pixels. Larger = more detail but slower. 384-512 recommended."}),
                 "max_size": ("INT", {"default": 1024, "min": 64, "max": 2048, "advanced": True, "tooltip": "Maximum allowed face crop. Prevents excessive VRAM usage on very large faces."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "face_detail"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def face_detail(self, gen_pipe, bbox_detector, denoise, enabled=True, guide_size=512, max_size=1024):
        image = gen_pipe.image
        if image is None:
            raise ValueError("FaceDetailer: No image in pipeline.")
        if not enabled: return (gen_pipe,)

        pp = extract_pipeline_params(gen_pipe)
        positive, negative = encode_prompts(pp.clip, pp.pos_text, pp.neg_text)

        segs = bbox_detector.detect(image, 0.5, 10, 3.0, 10)

        result = fd_logic.do_detail(
                 image=image, segs=segs, model=pp.model, clip=pp.clip, vae=pp.vae,
                 guide_size=guide_size, guide_size_for_bbox=True, max_size=max_size,
                 seed=pp.seed, steps=pp.steps, cfg=pp.cfg, sampler_name=pp.sampler_name, scheduler=pp.scheduler,
                 positive=positive, negative=negative, denoise=denoise,
                 feather=5, noise_mask=True, force_inpaint=True, drop_size=10
             )
        ctx = gen_pipe.clone()
        ctx.image = result[0]
        return (ctx,)
