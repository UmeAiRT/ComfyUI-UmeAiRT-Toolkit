"""
UmeAiRT Toolkit - Subject Detailer Nodes
------------------------------------------
Pipeline-aware Subject Detailer node with manifest-driven auto-downloads
for BBOX detectors (faces, hands, etc.).
"""

from .common import log_node, encode_prompts, extract_pipeline_params
from .manifest import download_bundle_files

try:
    from ..facedetailer_core import logic as fd_logic
    from ..facedetailer_core import detector
except ImportError as e:
    log_node(f"Subject Nodes: Could not import FaceDetailer internals: {e}", color="YELLOW")


# --- Pipeline-Aware Subject Detailer ---

class UmeAiRT_PipelineSubjectDetailer:
    """Subject detailer — automatically fetches detector models and acts on the pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                 "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                 "subject": (["face", "hand", "both"], {"default": "face", "tooltip": "Which subject to automatically detect and detail. 'both' processes faces then hands sequentially."}),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
                 
                 # Advanced section
                 "guide_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8, "advanced": True, "tooltip": "Target face crop size in pixels. Larger = more detail but slower. 384-512 recommended."}),
                 "max_size": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8, "advanced": True, "tooltip": "Maximum allowed face crop. Prevents excessive VRAM usage on very large faces."}),
                 "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Detection confidence. Higher = fewer false positives, but might miss faces."}),
                 "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1, "advanced": True, "tooltip": "Expands the detection mask outwards in pixels."}),
                 "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1, "advanced": True, "tooltip": "How much surrounding context to include when cropping the face."}),
                 "drop_size": ("INT", {"default": 10, "min": 1, "max": 256, "step": 1, "advanced": True, "tooltip": "Ignore faces smaller than this many pixels."}),
                 "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "advanced": True, "tooltip": "Blend softness of the restored face edges."}),
                 "noise_mask": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Apply noise mask during inpainting. Usually best kept On."}),
                 "force_inpaint": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Force the masking to behave like an inpaint."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "subject_detail"
    CATEGORY = "UmeAiRT/Pipeline/Post-Processing"

    def subject_detail(self, gen_pipe, subject, denoise, enabled=True, 
                    guide_size=512, max_size=1024, bbox_threshold=0.5, bbox_dilation=10, 
                    bbox_crop_factor=3.0, drop_size=10, feather=5, noise_mask=True, force_inpaint=True):
        image = gen_pipe.image
        if image is None:
            raise ValueError("Subject Detailer: No image in pipeline.")
        if not enabled: return (gen_pipe,)

        pp = extract_pipeline_params(gen_pipe)
        positive, negative = encode_prompts(pp.clip, pp.pos_text, pp.neg_text)

        current_image = image
        subjects_to_process = ["face", "hand"] if subject == "both" else [subject]

        for subj in subjects_to_process:
            # Auto-download the requested bbox model
            try:
                resolved_files, meta, dn, sk, err = download_bundle_files("_BBOX_MODELS", subj)
                if err:
                    raise RuntimeError(f"Subject Detailer: Failed to auto-download {subj} detector: {', '.join(err)}")
                bbox_filename = resolved_files["bbox"][0]
            except Exception as e:
                log_node(f"Subject Detailer: Manifest resolution error for '{subj}': {e}", color="RED")
                raise RuntimeError(f"Subject Detailer: failed to retrieve detector for '{subj}': {e}")

            # Load the model directly
            bbox_detector_model = detector.load_bbox_model(bbox_filename)

            segs = bbox_detector_model.detect(current_image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

            result = fd_logic.do_detail(
                     image=current_image, segs=segs, model=pp.model, clip=pp.clip, vae=pp.vae,
                     guide_size=guide_size, guide_size_for_bbox=True, max_size=max_size,
                     seed=pp.seed, steps=pp.steps, cfg=pp.cfg, sampler_name=pp.sampler_name, scheduler=pp.scheduler,
                     positive=positive, negative=negative, denoise=denoise,
                     feather=feather, noise_mask=noise_mask, force_inpaint=force_inpaint, drop_size=drop_size
                 )
            current_image = result[0]
             
        ctx = gen_pipe.clone()
        ctx.image = current_image
        return (ctx,)
