"""
UmeAiRT Toolkit - Image Nodes
-------------------------------
Image I/O nodes migrated to use GenerationContext pipeline.
"""

import torch
import numpy as np
import os
import re
import random
import string
import folder_paths
import comfy.utils
import nodes as comfy_nodes
import torchvision.transforms.functional as TF
from .common import resize_tensor, apply_outpaint_padding, log_node
from .logger import logger
from .image_saver_core.logic import ImageSaverLogic



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
    CATEGORY = "UmeAiRT/Pipeline/Output"

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
