"""
UmeAiRT Toolkit - Detailer Daemon Nodes
-----------------------------------------
Pipeline-aware Detail Daemon nodes (Simple & Advanced).
"""

import torch
import numpy as np
import comfy.samplers
import comfy.sample
from .common import log_node, encode_prompts, extract_pipeline_params


# --- Detail Daemon Schedule Helpers ---

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


# --- Detailer Daemon Nodes ---

class UmeAiRT_Detailer_Daemon:
    """Detail daemon — enhances fine details by modifying the sampling schedule.

    Reads models/settings from pipeline. Schedule parameters are hidden
    behind 'Show advanced inputs' by default.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline carrying your image, model, and all settings through the workflow."}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Passthrough", "tooltip": "Turn this effect on or off. When off, the image passes through unchanged."}),
                "detail_amount": ("FLOAT", {"default": 0.5, "min": -5.0, "max": 5.0, "step": 0.01, "display": "slider", "tooltip": "How much extra detail to add. Positive = sharpen, negative = soften. Start with 0.1-0.5."}),
                "start": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "When detail enhancement begins in the sampling process (0.0 = from the start)."}),
                "end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "When detail enhancement ends in the sampling process (1.0 = until completion)."}),
                "bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Shifts the peak of detail enhancement within the start-end range."}),
                "exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "advanced": True, "tooltip": "Controls how sharply the detail effect ramps up/down. Higher = more concentrated."}),
                "start_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Fine-tune the exact starting point of the detail curve."}),
                "end_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "Fine-tune the exact ending point of the detail curve."}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True, "tooltip": "Gradually reduce the effect at the edges of the schedule range."}),
                "smooth": ("BOOLEAN", {"default": True, "advanced": True, "tooltip": "Smooth out the detail schedule curve to avoid abrupt changes."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "How much the AI redraws during upscale. Lower = sharper but less detail added."}),
                "refine_denoise": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True, "tooltip": "How much to redraw in the refinement pass. Lower = preserves more original detail."}),
            },
            "optional": {
                "steps": ("INT", {"default": 20, "advanced": True, "tooltip": "Total sampling steps. More steps = better quality but slower. 20-30 recommended."}),
                "refine_steps": ("INT", {"default": 2, "advanced": True, "tooltip": "Steps for the extra refinement pass. 5-10 usually sufficient."}),
                "cfg": ("FLOAT", {"default": 8.0, "advanced": True, "tooltip": "How closely the AI follows your prompt. Higher = more literal but may look artificial. 5-8 recommended."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"advanced": True}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"advanced": True}),
                "seed": ("INT", {"default": 0, "advanced": True, "tooltip": "Seed for reproducible results. Same seed + same settings = same image."}),
            }
        }

    RETURN_TYPES = ("UME_PIPELINE",)
    RETURN_NAMES = ("gen_pipe",)
    FUNCTION = "process"
    CATEGORY = "UmeAiRT/Post-Process"

    def process(self, gen_pipe, enabled, detail_amount,
                start=0.2, end=0.8, bias=0.5, exponent=1.0,
                start_offset=0.0, end_offset=0.0, fade=0.0, smooth=True,
                denoise=0.5, refine_denoise=0.05,
                steps=20, refine_steps=2, cfg=8.0, sampler_name="euler", scheduler="normal", seed=0):
        if not enabled:
            return (gen_pipe,)

        start_image = gen_pipe.image
        if start_image is None:
            raise ValueError("Detail Daemon: No image in pipeline.")

        pp = extract_pipeline_params(gen_pipe)
        model, vae, clip = pp.model, pp.vae, pp.clip

        positive, negative = encode_prompts(clip, pp.pos_text, pp.neg_text)

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
        log_node("Detail Daemon: Finished", color="GREEN")
        ctx = gen_pipe.clone()
        ctx.image = decoded
        return (ctx,)

