"""
UmeAiRT Toolkit - Image Nodes
-------------------------------
Image I/O nodes migrated to use GenerationContext pipeline.
"""

import torch
import numpy as np
import os
import re
import folder_paths
import comfy.utils
import nodes as comfy_nodes
import torchvision.transforms.functional as TF
from .common import resize_tensor, apply_outpaint_padding, log_node
from .logger import logger
from .image_saver_core.logic import ImageSaverLogic


class UmeAiRT_PipelineImageLoader(comfy_nodes.LoadImage):
    """Image Loader with optional resize from pipeline dimensions."""
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Pipeline", "label_off": "Keep Original", "tooltip": "Resize the image to match the generation pipeline dimensions."}),
                "mode": (["Inpaint", "Img2Img"], {"default": "Inpaint", "tooltip": "Inpaint: fills the masked area. Img2Img: transforms the whole image (ignores mask)."}),
            },
            "optional": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "Connect the generation pipeline to automatically use its width/height for resizing."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image_wireless"
    CATEGORY = "UmeAiRT/Pipeline/Image"

    def load_image_wireless(self, image, resize, mode, gen_pipe=None):
        out = super().load_image(image)
        img = out[0]
        mask = out[1]

        if resize and gen_pipe is not None:
            target_w = int(gen_pipe.width or 1024)
            target_h = int(gen_pipe.height or 1024)
            img = resize_tensor(img, target_h, target_w, interp_mode="bilinear", is_mask=False)
            if mask is not None:
                mask = resize_tensor(mask, target_h, target_w, interp_mode="nearest", is_mask=True)

        if mode == "Img2Img":
            mask = None

        return (img, mask)


class UmeAiRT_SourceImage_Output:
    """Passthrough node for source image/mask. No longer uses global state."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE", {"tooltip": "The original source image to pass through."}),
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "Optional mask for the source image (white = areas to modify)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("source_image", "source_mask")
    FUNCTION = "get_source"
    CATEGORY = "UmeAiRT/Pipeline/Image"

    def get_source(self, source_image, source_mask=None):
        if source_mask is None:
             source_mask = torch.zeros((1, source_image.shape[1], source_image.shape[2]), dtype=torch.float32, device="cpu")
        return (source_image, source_mask)


class UmeAiRT_PipelineImageProcess:
    """Image pre-processing node (resize, pad, blur mask) — reads dimensions from pipeline."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "How much the AI changes the image. 1.0 = completely new, 0.3 = subtle changes only."}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img", "tooltip": "How to process the image: img2img (transform), inpaint (fill masked area), or outpaint (extend)."}),
            },
            "optional": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "Connect the generation pipeline to use its width/height for resizing."}),
                "image": ("IMAGE", {"tooltip": "The image to process (from a loader or previous node)."}),
                "mask": ("MASK", {"tooltip": "Optional mask for inpainting (white = areas to modify)."}),
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Resize the image to match the generation pipeline dimensions."}),
                "mask_blur": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": "Softens the mask edge for natural blending. Higher values = smoother transitions."}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels to extend the image on the left (outpaint mode)."}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels to extend the image on the top (outpaint mode)."}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels to extend the image on the right (outpaint mode)."}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "Pixels to extend the image on the bottom (outpaint mode)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Pipeline/Image"

    def process_image(self, denoise=1.0, mode="img2img", gen_pipe=None, image=None, mask=None, resize=False, mask_blur=0,
                      padding_left=0, padding_top=0, padding_right=0, padding_bottom=0):

        if mode == "txt2img":
             log_node("ImageProcess: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             return (image, None)

        if image is None:
            return (None, None)

        B, H, W, C = image.shape

        target_w, target_h = W, H
        if resize and gen_pipe is not None:
             target_w = int(gen_pipe.width or 1024)
             target_h = int(gen_pipe.height or 1024)

        final_image = image
        final_mask = mask

        if resize:
             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear", is_mask=False)
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)
             B, H, W, C = final_image.shape

        # OUTPAINT
        if mode == "outpaint":
             pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
             final_image, final_mask = apply_outpaint_padding(
                 final_image, final_mask, pad_l, pad_t, pad_r, pad_b, overlap=8, feathering=40
             )

        # INPAINT BLUR
        if mask_blur > 0 and final_mask is not None:
             if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
             elif len(final_mask.shape) == 3: m = final_mask.unsqueeze(1)
             else: m = final_mask

             k = mask_blur
             if k % 2 == 0: k += 1
             m = TF.gaussian_blur(m, kernel_size=k)

             if len(final_mask.shape) == 2: final_mask = m.squeeze(0).squeeze(0)
             elif len(final_mask.shape) == 3: final_mask = m.squeeze(1)

        if mode == "img2img":
            final_mask = None
            log_node("ImageProcess: Img2Img Mode (Mask Hidden).", color="YELLOW")

        return (final_image, final_mask)


class UmeAiRT_PipelineInpaintComposite:
    """Composites generated image back onto source using mask."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generated_image": ("IMAGE", {"tooltip": "The AI-generated image to composite onto the source."}),
                "source_image": ("IMAGE", {"tooltip": "The original image to blend the generated result onto."}),
            },
            "optional": {
                "source_mask": ("MASK", {"tooltip": "Black/white mask defining which areas to keep (black) vs replace (white)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    FUNCTION = "composite"
    CATEGORY = "UmeAiRT/Pipeline/Image"

    def composite(self, generated_image, source_image, source_mask=None):
        if source_mask is None:
            log_node("InpaintComposite: No mask provided, returning generated image.", color="YELLOW")
            return (generated_image,)

        gB, gH, gW, gC = generated_image.shape
        sB, sH, sW, sC = source_image.shape

        source_resized = source_image
        if sH != gH or sW != gW:
            source_resized = resize_tensor(source_image, gH, gW, interp_mode="bilinear", is_mask=False)

        mask_resized = source_mask
        if len(mask_resized.shape) == 2:
            mH, mW = mask_resized.shape
        elif len(mask_resized.shape) == 3:
             mB, mH, mW = mask_resized.shape
        else:
            mH, mW = gH, gW

        if mH != gH or mW != gW:
             mask_resized = resize_tensor(source_mask, gH, gW, interp_mode="bilinear", is_mask=True)

        m = mask_resized
        if len(m.shape) == 2:
            m = m.unsqueeze(0).unsqueeze(-1)
        elif len(m.shape) == 3:
            m = m.unsqueeze(-1)

        if m.shape[0] < gB:
            m = m.repeat(gB, 1, 1, 1)
        if source_resized.shape[0] < gB:
            source_resized = source_resized.repeat(gB, 1, 1, 1)

        composite = source_resized * (1.0 - m) + generated_image * m

        return (composite,)


class UmeAiRT_PipelineImageSaver:
    """Image saver with metadata from pipeline context."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gen_pipe": ("UME_PIPELINE", {"tooltip": "The generation pipeline containing the AI-generated image and all metadata."}),
                "filename": ("STRING", {"default": "%date%_%time%_%model%_%seed%", "multiline": False, "tooltip": "Filename for saved images. Use variables: %date%, %time%, %model%, %seed%, %width%, %height%."}),
            },
            "hidden": {
                 "prompt": "PROMPT",
                 "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "UmeAiRT/Pipeline/IO"

    def save_images(self, gen_pipe, filename, prompt=None, extra_pnginfo=None):
        images = gen_pipe.image
        if images is None:
            raise ValueError("Image Saver: No image in pipeline.")
        while ".." in filename:
            filename = filename.replace("..", "")

        full_pattern = filename.replace("\\", "/")
        if "/" in full_pattern:
             path, filename = full_pattern.rsplit("/", 1)
        else:
             path = ""
             filename = full_pattern

        path = path.lstrip("/\\")
        path = re.sub(r'[<>:"\\|?*]', '', path)
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)

        extension = "png"
        lossless_webp = True
        quality_jpeg_or_webp = 100
        optimize_png = False
        embed_workflow = True
        save_workflow_as_json = False

        # Read from generation pipeline
        width = int(gen_pipe.width or 512)
        height = int(gen_pipe.height or 512)
        modelname = getattr(gen_pipe, 'model_name', 'UmeAiRT_Pipeline')

        additional_hashes = ""
        loras = gen_pipe.loras or []
        if loras:
            try:
                from .image_saver_core.utils import full_lora_path_for, get_sha256
                hash_list = []
                for lora in loras:
                    name = lora.get("name")
                    strength = lora.get("strength", 1.0)
                    if name:
                        path_to_lora = full_lora_path_for(name)
                        if path_to_lora:
                            l_hash = get_sha256(path_to_lora)[:10]
                            hash_list.append(f"{name}:{l_hash}:{strength}")
                if hash_list:
                    additional_hashes = ",".join(hash_list)
            except Exception as e:
                log_node(f"Error processing LoRAs for metadata: {e}", color="RED")

        try:
            metadata_obj = ImageSaverLogic.make_metadata(
                modelname=modelname,
                positive=str(gen_pipe.positive_prompt or ""),
                negative=str(gen_pipe.negative_prompt or ""),
                width=width,
                height=height,
                seed_value=int(gen_pipe.seed or 0),
                steps=int(gen_pipe.steps or 20),
                cfg=float(gen_pipe.cfg or 8.0),
                sampler_name=gen_pipe.sampler_name or "euler",
                scheduler_name=gen_pipe.scheduler or "normal",
                denoise=float(gen_pipe.denoise or 1.0),
                clip_skip=0,
                custom="UmeAiRT Pipeline",
                additional_hashes=additional_hashes,
                download_civitai_data=False,
                easy_remix=True
            )
        except Exception as e:
            log_node(f"Metadata Creation Failed: {e}", color="RED")
            raise e

        time_format = "%Y-%m-%d-%H%M%S"

        import random
        import string
        rand_suffix = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        filename = f"{filename}_{rand_suffix}"

        resolved_path = ImageSaverLogic.replace_placeholders(
            path,
            metadata_obj.width, metadata_obj.height, metadata_obj.seed, metadata_obj.modelname,
            getattr(self, "counter", 0), time_format,
            metadata_obj.sampler_name, metadata_obj.steps, metadata_obj.cfg, metadata_obj.scheduler_name,
            metadata_obj.denoise, metadata_obj.clip_skip, metadata_obj.custom
        )

        resolved_path = resolved_path.lstrip("/\\")
        output_dir_abs = os.path.abspath(folder_paths.output_directory)
        final_abs_path = os.path.abspath(os.path.join(output_dir_abs, resolved_path))

        if not final_abs_path.startswith(output_dir_abs):
             log_node(f"Security Warning: Path Traversal blocked. Attempted: {final_abs_path}", color="RED")
             resolved_path = ""

        try:
            if not hasattr(self, "counter"):
                self.counter = 0

            result_filenames = ImageSaverLogic.save_images(
                images=images,
                filename_pattern=filename,
                extension=extension,
                path=resolved_path,
                quality_jpeg_or_webp=quality_jpeg_or_webp,
                lossless_webp=lossless_webp,
                optimize_png=optimize_png,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                save_workflow_as_json=save_workflow_as_json,
                embed_workflow=embed_workflow,
                counter=self.counter,
                time_format=time_format,
                metadata=metadata_obj
            )

            self.counter += len(images)

            if len(result_filenames) == 1:
                log_node(f"Image Saver: Saved -> {resolved_path}/{result_filenames[0]}", color="GREEN")
            else:
                log_node(f"Image Saver: Saved {len(result_filenames)} images -> {resolved_path}", color="GREEN")

            ui_images = []
            for fname in result_filenames:
                ui_images.append({
                    "filename": fname,
                    "subfolder": resolved_path,
                    "type": "output"
                })
            return {"ui": {"images": ui_images}}

        except Exception as e:
            log_node(f"Save Failed: {e}", color="RED")
            return {"ui": {"images": []}}
