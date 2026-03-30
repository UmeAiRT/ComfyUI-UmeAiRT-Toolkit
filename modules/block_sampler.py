import torch
import torchvision.transforms.functional as TF
import weakref
import folder_paths
import nodes as comfy_nodes
import comfy.sd
import comfy.utils
from .common import GenerationContext, resize_tensor, apply_outpaint_padding, log_node, validate_bundle
from .logger import logger
from .optimization_utils import SamplerContext, warmup_vae
from typing import Tuple, Dict, Any, Optional, List

try:
    from .facedetailer_core import detector, logic as fd_logic
except ImportError:
    pass

# --- Processor Blocks ---

class UmeAiRT_BlockSampler:
    """Central hub: receives models + settings + prompts as side-inputs,
    creates the GenerationContext pipeline, samples, and stores the image inside.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("UME_BUNDLE", {"tooltip": "Model bundle from a Loader node."}),
                "positive": ("POSITIVE", {"forceInput": True, "tooltip": "Describe what you want in the image. Connect a Prompt Input or CLIP Text Encode node."}),
                "negative": ("NEGATIVE", {"forceInput": True, "tooltip": "Describe what to avoid in the image. Connect a Prompt Input or CLIP Text Encode node."}),
                "settings": ("UME_SETTINGS", {"tooltip": "Settings from Generation Settings node."}),
            },
            "optional": {
                "loras": ("UME_LORA_STACK", {"tooltip": "Connect a LoRA Block node to apply style/character modifications to the model."}),
                "images": ("UME_IMAGE", {"tooltip": "Connect an Image Process node for img2img, inpaint, or outpaint workflows."}),
            }
        }
    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Block/Sampler"

    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()
        self.cnet_loader = comfy_nodes.ControlNetLoader()
        self.cnet_apply = comfy_nodes.ControlNetApplyAdvanced()
        self._vae_encode = comfy_nodes.VAEEncode()
        self._ksampler = comfy_nodes.KSampler()
        self._vae_decode = comfy_nodes.VAEDecode()
        self._controlnet_cache = {}
        self._last_pos_text = None
        self._last_neg_text = None
        self._last_clip_ref = None
        self._cached_positive = None
        self._cached_negative = None
        self._last_loras = None
        self._last_controlnets = None

    def process(self, 
                model_bundle: Dict[str, Any], 
                positive: Optional[str] = None, 
                negative: Optional[str] = None,
                settings: Dict[str, Any] = None, 
                loras: Optional[List[Tuple[str, float, float]]] = None, 
                images: Optional[Dict[str, Any]] = None) -> Tuple[GenerationContext]:
        # 1. Validate and unpack model_bundle
        validate_bundle(model_bundle, ["model", "clip", "vae"], context="Block Sampler")
        model = model_bundle.model
        clip = model_bundle.clip
        vae = model_bundle.vae

        ctx = GenerationContext()
        ctx.model = model
        ctx.clip = clip
        ctx.vae = vae
        ctx.model_name = model_bundle.model_name

        # 2. Apply settings
        ctx.width = settings.width
        ctx.height = settings.height
        ctx.steps = settings.steps
        ctx.cfg = settings.cfg
        ctx.sampler_name = settings.sampler_name
        ctx.scheduler = settings.scheduler
        ctx.seed = settings.seed

        controlnets = []
        if images:
            controlnets = images.controlnets if images.controlnets else []

        # 3. Apply LoRAs
        if loras:
            if not model or not clip: raise ValueError("Block Sampler: No base Model/CLIP for LoRAs.")
            loaded_loras_meta = []
            for lora_def in loras:
                name, str_model, str_clip = lora_def
                if name != "None":
                    try:
                         model, clip = self.lora_loader.load_lora(model, clip, name, str_model, str_clip)
                         loaded_loras_meta.append({"name": name, "strength": str_model})
                    except Exception as e:
                        log_node(f"Block Sampler LoRA Error ({name}): {e}", color="RED")
            ctx.model = model
            ctx.clip = clip
            ctx.loras = loaded_loras_meta

        if not model or not vae or not clip: raise ValueError("Block Sampler: Missing Model/VAE/CLIP.")

        width, height = ctx.width, ctx.height
        steps, cfg = ctx.steps, ctx.cfg
        sampler_name, scheduler = ctx.sampler_name, ctx.scheduler
        seed = ctx.seed

        denoise = images.denoise if images else ctx.denoise
        ctx.denoise = denoise

        # 4. Handle Prompts
        pos_text = positive if positive is not None else ""
        neg_text = negative if negative is not None else ""
        ctx.positive_prompt = pos_text
        ctx.negative_prompt = neg_text

        latent_image = None
        mode_str = "txt2img"
        raw_image, source_mask = None, None

        if images:
             raw_image = images.image
             source_mask = images.mask
             mode_str = images.mode or "img2img"
             auto_resize = images.auto_resize

             # Auto-resize source image to settings dimensions
             if auto_resize and raw_image is not None:
                 raw_image = resize_tensor(raw_image, height, width, interp_mode="bilinear")
                 if source_mask is not None:
                     source_mask = resize_tensor(source_mask, height, width, interp_mode="nearest", is_mask=True)

             # Outpaint: resize source to fit within target, then pad
             if mode_str == "outpaint" and raw_image is not None:
                 target_w = images.outpaint_target_w
                 target_h = images.outpaint_target_h
                 h_align = images.outpaint_h_align
                 v_align = images.outpaint_v_align
                 mask_blur = images.outpaint_mask_blur

                 B, src_h, src_w, C = raw_image.shape

                 # Resize source to fit within target (maintain aspect ratio)
                 if src_w > target_w or src_h > target_h:
                     scale = min(target_w / src_w, target_h / src_h)
                     fit_w = int(src_w * scale)
                     fit_h = int(src_h * scale)
                     raw_image = resize_tensor(raw_image, fit_h, fit_w, interp_mode="bilinear")
                     if source_mask is not None:
                         source_mask = resize_tensor(source_mask, fit_h, fit_w, interp_mode="nearest", is_mask=True)
                     src_w, src_h = fit_w, fit_h

                 # Compute padding from alignment
                 total_pad_w = max(0, target_w - src_w)
                 total_pad_h = max(0, target_h - src_h)

                 if h_align == "left":
                     pad_l, pad_r = 0, total_pad_w
                 elif h_align == "right":
                     pad_l, pad_r = total_pad_w, 0
                 else:  # center
                     pad_l = total_pad_w // 2
                     pad_r = total_pad_w - pad_l

                 if v_align == "top":
                     pad_t, pad_b = 0, total_pad_h
                 elif v_align == "bottom":
                     pad_t, pad_b = total_pad_h, 0
                 else:  # center
                     pad_t = total_pad_h // 2
                     pad_b = total_pad_h - pad_t

                 # Apply padding + mask generation
                 raw_image, source_mask = apply_outpaint_padding(
                     raw_image, source_mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40
                 )

                 # Blur the mask
                 if source_mask is not None and mask_blur > 0:
                     if len(source_mask.shape) == 2:
                         m = source_mask.unsqueeze(0).unsqueeze(0)
                     else:
                         m = source_mask
                     k = mask_blur
                     if k % 2 == 0:
                         k += 1
                     m = TF.gaussian_blur(m, kernel_size=k)
                     source_mask = m.squeeze(0).squeeze(0) if len(source_mask.shape) == 2 else m

                 mode_str = "inpaint"

             ctx.source_image = raw_image
             ctx.source_mask = source_mask

             if mode_str in ["inpaint", "outpaint"] and source_mask is not None:
                 latent_image = self._vae_encode.encode(vae, raw_image)[0]
                 latent_image["noise_mask"] = source_mask
             elif denoise < 1.0:
                 latent_image = self._vae_encode.encode(vae, raw_image)[0]

        if latent_image is None and ctx.latent is not None:
             latent_image = ctx.latent

        # 5. Empty Latent — detect channels from model (SD=4, FLUX=16)
        if latent_image is None:
             latent_channels = 4
             try:
                 latent_channels = model.model.latent_format.latent_channels
             except Exception as e:
                 log_node(f"Block Sampler: Could not detect latent channels, defaulting to 4: {e}", color="YELLOW")
             l = torch.zeros([1, latent_channels, height // 8, width // 8], device="cpu")
             latent_image = {"samples": l}
             denoise = 1.0

        # 6. Encode prompts
        # Cache Strategy: If the prompt text matches, the CLIP object matches, and the exact same 
        # Modifiers (LoRAs/ControlNets configurations) are active, it is safe to use the cache.
        
        last_clip = self._last_clip_ref() if self._last_clip_ref is not None else None
        can_use_cache = (
            self._last_pos_text == pos_text and 
            self._last_neg_text == neg_text and 
            last_clip is clip and
            self._last_loras == loras and
            self._last_controlnets == controlnets
        )

        if can_use_cache:
             positive_cond = self._cached_positive
             negative_cond = self._cached_negative
             log_node("Block Sampler: Using cached Prompts (Fast Start)", color="GREEN")
        else:
             log_node("Block Sampler: Encoding Prompts...")
             tokens = clip.tokenize(pos_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             positive_cond = [[cond, {"pooled_output": pooled}]]

             tokens = clip.tokenize(neg_text)
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             negative_cond = [[cond, {"pooled_output": pooled}]]

             self._last_pos_text = pos_text
             self._last_neg_text = neg_text
             import weakref
             self._last_clip_ref = weakref.ref(clip)
             self._last_loras = loras
             self._last_controlnets = controlnets
             self._cached_positive = positive_cond
             self._cached_negative = negative_cond

        if controlnets:
            for cnet_def in controlnets:
                c_name, c_image, c_str, c_start, c_end = cnet_def
                if c_name != "None" and c_image is not None:
                    try:
                        if c_name in self._controlnet_cache:
                            c_model = self._controlnet_cache[c_name]
                        else:
                            c_model = self.cnet_loader.load_controlnet(c_name)[0]
                            self._controlnet_cache[c_name] = c_model
                        positive_cond, negative_cond = self.cnet_apply.apply_controlnet(positive_cond, negative_cond, c_model, c_image, c_str, c_start, c_end)
                    except Exception as e: log_node(f"Block Sampler ControlNet Error: {e}", color="RED")

        log_node(f"Block Sampler: {mode_str} | {width}x{height} | Steps: {steps} | CFG: {cfg}")

        warmup_vae(vae, latent_image)

        # 7. Sample
        try:
             with SamplerContext():
                 result_latent = self._ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive_cond, negative_cond, latent_image, denoise)[0]
        except Exception as e:
             raise RuntimeError(f"Sampling Failed: {e}")

        # 8. Decode & store image in pipeline
        log_node("Block Sampler: Decoding VAE")
        image_out = self._vae_decode.decode(vae, result_latent)[0]

        if mode_str == "inpaint" and raw_image is not None and source_mask is not None:
             try:
                 B, H, W, C = image_out.shape
                 source_resized = resize_tensor(raw_image, H, W, interp_mode="bilinear")
                 mask_resized = resize_tensor(source_mask, H, W, interp_mode="bilinear", is_mask=True)
                 m = mask_resized
                 if len(m.shape) == 2: m = m.unsqueeze(0).unsqueeze(-1)
                 elif len(m.shape) == 3: m = m.unsqueeze(-1)
                 if m.shape[0] < B: m = m.repeat(B, 1, 1, 1)
                 if source_resized.shape[0] < B: source_resized = source_resized.repeat(B, 1, 1, 1)
                 m = torch.clamp(m, 0.0, 1.0)
                 image_out = source_resized * (1.0 - m) + image_out * m
                 log_node("Block Inpaint: Auto-Composited.", color="GREEN")
             except Exception as e:
                 log_node(f"Block Inpaint Composite Failed: {e}", color="RED")

        ctx.image = image_out
        ctx.latent = result_latent
        return (ctx,)

