import torch
import comfy.samplers
import comfy.sample
import nodes
from collections import namedtuple
from . import utils
import logging

# Define SEG locally to match expected structure
try:
    from ..logger import log_node
except ImportError:
    def log_node(msg, color=None, prefix="UmeAiRT"):
        print(f"[UmeAiRT] {msg}")

SEG = namedtuple("SEG",

                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    # Replicates nodes.common_ksampler logic to ensure standalone capability
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    pbar = comfy.utils.ProgressBar(steps)
    k_callback = None
    
    # We can pass a callback to update pbar if strict adherence to comfy is needed, 
    # but for embedded loop often we want silent or standard pbar.
    def callback(step, x0, x, total_steps):
        pbar.update_absolute(step + 1, total_steps)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    
    out = latent.copy()
    out["samples"] = samples
    return out


def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   noise_mask_feather=0):
    
    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip if bbox is already large enough
    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        return None

    if guide_size_for_bbox:
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            return None
        if new_w == 0 or new_h == 0:
            return None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            upscale = 1.0
            new_w = w
            new_h = h

    # Upscale
    upscaled_image = utils.tensor_resize(image, new_w, new_h)

    # Prepare mask for inpainting
    if noise_mask is not None:
        noise_mask = utils.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

    # Encode to latent
    latent_image = utils.to_latent_image(upscaled_image, vae)
    if noise_mask is not None:
        latent_image['noise_mask'] = noise_mask

    # K-Sample
    # NOTE: We assume 'model' is a standard comfy model object
    refined_latent = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

    # Decode
    refined_image = vae.decode(refined_latent['samples'])
    
    # Resize back to original crop size
    refined_image = utils.tensor_resize(refined_image, w, h)
    
    return refined_image


def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name, scheduler,
              positive, negative, denoise, feather, noise_mask, force_inpaint, noise_mask_feather=0, drop_size=10):

    # SEGS structure: (shape, [seg1, seg2, ...])
    # segs[1] contains the list of SEG objects
    if not segs or len(segs) < 2:
        return (image,)

    # We iterate over the list of segments
    ordered_segs = segs[1]
    
    enhanced_image_result = image.clone()

    for i, seg in enumerate(ordered_segs):
        # seg is an object (likely impact.core.SEG). We access attributes assuming they exist.
        cropped_image = utils.crop_ndarray4(enhanced_image_result.cpu().numpy(), seg.crop_region)
        cropped_image = utils.to_tensor(cropped_image)
        
        mask = utils.to_tensor(seg.cropped_mask)
        mask = utils.tensor_gaussian_blur_mask(mask, feather)

        is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
        if is_mask_all_zeros:
            continue

        if noise_mask:
            cropped_mask = seg.cropped_mask
        else:
            cropped_mask = None

        # Calculate seed for this segment
        seg_seed = seed + i
        
        try:
            enhanced_patch = enhance_detail(
                cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
                seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, denoise, cropped_mask, force_inpaint,
                noise_mask_feather
            )
        except Exception as e:
            # print(f"Error enhancing segment {i}: {e}")
            log_node(f"FaceDetailer Error enhancing segment {i}: {e}", color="RED")
            enhanced_patch = None


        if enhanced_patch is not None:
             # Ensure devices match before pasting
            enhanced_image_result = enhanced_image_result.cpu()
            enhanced_patch = enhanced_patch.cpu()
            
            # The mask needs to be resized to match the enhanced patch if logic changed, 
            # but usually enhance_detail returns same size as input cropped_image.
            # However, if dimensions changed slightly due to resizing, we must ensure mask matches.
            
            utils.tensor_paste(enhanced_image_result, enhanced_patch, (seg.crop_region[0], seg.crop_region[1]), mask)

    return (utils.tensor_convert_rgb(enhanced_image_result),)
