import os
import json
import re
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import folder_paths
from .saver import save_image
from .utils import sanitize_filename, get_sha256, full_checkpoint_path_for
from .utils_civitai import get_civitai_sampler_name, get_civitai_metadata, MAX_HASH_LENGTH
from .prompt_metadata_extractor import PromptMetadataExtractor

try:
    from ..logger import log_node
except ImportError:
    def log_node(msg, color=None, prefix="UmeAiRT"):
        print(f"[UmeAiRT] {msg}")


def parse_checkpoint_name(ckpt_name: str) -> str:
    return os.path.basename(ckpt_name)

def parse_checkpoint_name_without_extension(ckpt_name: str) -> str:
    filename = parse_checkpoint_name(ckpt_name)
    name_without_ext, ext = os.path.splitext(filename)
    supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}

    if ext.lower() in supported_extensions:
        return name_without_ext
    else:
        return filename

def get_timestamp(time_format: str) -> str:
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except Exception:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")
    return timestamp

def apply_custom_time_format(filename: str) -> str:
    now = datetime.now()
    pattern = r'%time_format<([^>]*)>'
    def replace_format(match):
        format_str = match.group(1)
        try:
            return now.strftime(format_str)
        except Exception:
            return match.group(0)
    return re.sub(pattern, replace_format, filename)

def save_json(image_info: dict[str, Any] | None, filename: str) -> None:
    try:
        workflow = (image_info or {}).get('workflow')
        if workflow is None:
            # print('No image info found, skipping saving of JSON')
            log_node('ImageSaver No image info found, skipping JSON.', color="YELLOW")
        with open(f'{filename}.json', 'w') as workflow_file:

            json.dump(workflow, workflow_file)
            # print(f'Saved workflow to {filename}.json')
            log_node(f'ImageSaver Saved workflow to {filename}.json', color="GREEN")
    except Exception as e:

        # print(f'Failed to save workflow as json due to: {e}, proceeding with the remainder of saving execution')
        log_node(f'ImageSaver Failed to save JSON: {e}', color="RED")


@dataclass
class Metadata:
    modelname: str
    positive: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler_name: str
    denoise: float
    clip_skip: int
    custom: str
    additional_hashes: str
    ckpt_path: str
    a111_params: str
    final_hashes: str

class ImageSaverLogic:
    # Match 'anything' or 'anything:anything' with trimmed white space
    re_manual_hash = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?\s*$')
    # Match 'anything', 'anything:anything' or 'anything:anything:number' with trimmed white space
    re_manual_hash_weights = re.compile(r'^\s*([^:]+?)(?:\s*:\s*([^\s:][^:]*?))?(?:\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)))?\s*$')

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor.ndim == 4:
            tensor = tensor[0]
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    @staticmethod
    def replace_placeholders(text: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, clip_skip: int, custom: str) -> str:
        text = apply_custom_time_format(text)
        text = text.replace("%date", get_timestamp("%Y-%m-%d"))
        text = text.replace("%time", get_timestamp(time_format))
        text = text.replace("%model", parse_checkpoint_name(modelname))
        text = text.replace("%width", str(width))
        text = text.replace("%height", str(height))
        text = text.replace("%seed", str(seed))
        text = text.replace("%counter", str(counter))
        text = text.replace("%sampler_name", sampler_name)
        text = text.replace("%steps", str(steps))
        text = text.replace("%cfg", str(cfg))
        text = text.replace("%scheduler_name", scheduler_name)
        text = text.replace("%basemodelname", parse_checkpoint_name_without_extension(modelname))
        text = text.replace("%denoise", str(denoise))
        text = text.replace("%clip_skip", str(clip_skip))
        text = text.replace("%custom", custom)
        return text

    @staticmethod
    def make_pathname(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, clip_skip: int, custom: str) -> str:
        filename = ImageSaverLogic.replace_placeholders(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler_name, denoise, clip_skip, custom)
        directory, basename = os.path.split(filename)
        sanitized_basename = sanitize_filename(basename)
        return os.path.join(directory, sanitized_basename)

    @staticmethod
    def make_filename(filename: str, width: int, height: int, seed: int, modelname: str, counter: int, time_format: str, sampler_name: str, steps: int, cfg: float, scheduler_name: str, denoise: float, clip_skip: int, custom: str) -> str:
        filename = ImageSaverLogic.make_pathname(filename, width, height, seed, modelname, counter, time_format, sampler_name, steps, cfg, scheduler_name, denoise, clip_skip, custom)
        return get_timestamp(time_format) if filename == "" else filename

    @staticmethod
    def clean_prompt(prompt: str, metadata_extractor: PromptMetadataExtractor) -> str:
        prompt = re.sub(metadata_extractor.LORA, "", prompt)
        prompt = re.sub(metadata_extractor.EMBEDDING, lambda match: Path(match.group(1)).stem, prompt)
        prompt = re.sub(r'\b[A-Z]+\([^)]*\)', "", prompt)
        return prompt

    @staticmethod
    def get_multiple_models(modelname: str, additional_hashes: str) -> tuple[str, str]:
        model_names = [m.strip() for m in modelname.split(',')]
        modelname = model_names[0]
        for additional_model in model_names[1:]:
            additional_ckpt_path = full_checkpoint_path_for(additional_model)
            if additional_ckpt_path:
                additional_modelhash = get_sha256(additional_ckpt_path)[:10]
                if additional_hashes:
                    additional_hashes += ","
                additional_hashes += f"{additional_model}:{additional_modelhash}"
        return modelname, additional_hashes

    @staticmethod
    def parse_manual_hashes(additional_hashes: str, existing_hashes: set[str], download_civitai_data: bool) -> dict[str, tuple[str | None, float | None, str]]:
        manual_entries: dict[str, tuple[str | None, float | None, str]] = {}
        unnamed_count = 0
        additional_hash_split = additional_hashes.replace("\n", ",").split(",") if additional_hashes else []
        for entry in additional_hash_split:
            # Always use the more capable regex
            match = ImageSaverLogic.re_manual_hash_weights.search(entry)
            if match is None: continue
            
            groups = [g for g in match.groups() if g]
            
            # Extract weight if present (last element if strictly scalar lookalike)
            weight = None
            if len(groups) >= 3:
                # name, hash, weight
                try:
                    weight = float(groups[-1])
                    groups = groups[:-1]
                except (ValueError, TypeError): pass
            elif len(groups) == 2:
                 # Check if second item is number (implies name:weight, no hash? No regex expects hash.)
                 # Regex: group 1 (name/hash), group 2 (hash?), group 3 (weight).
                 # If we have 2 groups, it's name, hash.
                 pass

            # After popping weight if present:
            if len(groups) > 1:
                name, hash_val = groups[0], groups[1]
            else:
                name, hash_val = None, groups[0]

            if len(hash_val) > MAX_HASH_LENGTH: continue
            if any(hash_val.lower() == existing_hash.lower() for _, _, existing_hash in manual_entries.values()): continue
            if hash_val.lower() in existing_hashes: continue
            
            if name is None:
                unnamed_count += 1
                name = f"manual{unnamed_count}"
            manual_entries[name] = (None, weight, hash_val)
            if len(manual_entries) > 29: break
        return manual_entries

    @staticmethod
    def make_metadata(modelname: str, positive: str, negative: str, width: int, height: int, seed_value: int, steps: int, cfg: float, sampler_name: str, scheduler_name: str, denoise: float, clip_skip: int, custom: str, additional_hashes: str, download_civitai_data: bool, easy_remix: bool) -> Metadata:
        modelname, additional_hashes = ImageSaverLogic.get_multiple_models(modelname, additional_hashes)
        ckpt_path = full_checkpoint_path_for(modelname)
        modelhash = get_sha256(ckpt_path)[:10] if ckpt_path else ""

        metadata_extractor = PromptMetadataExtractor([positive, negative])
        embeddings = metadata_extractor.get_embeddings()
        loras = metadata_extractor.get_loras()
        civitai_sampler_name = get_civitai_sampler_name(sampler_name.replace('_gpu', ''), scheduler_name)
        basemodelname = parse_checkpoint_name_without_extension(modelname)

        existing_hashes = {modelhash.lower()} | {t[2].lower() for t in loras.values()} | {t[2].lower() for t in embeddings.values()}
        manual_entries = ImageSaverLogic.parse_manual_hashes(additional_hashes, existing_hashes, download_civitai_data)
        civitai_resources, hashes, add_model_hash = get_civitai_metadata(modelname, ckpt_path, modelhash, loras, embeddings, manual_entries, download_civitai_data)

        if easy_remix:
            positive = ImageSaverLogic.clean_prompt(positive, metadata_extractor)
            negative = ImageSaverLogic.clean_prompt(negative, metadata_extractor)

        positive_a111_params = positive.strip()
        negative_a111_params = f"\nNegative prompt: {negative.strip()}"
        clip_skip_str = f", Clip skip: {abs(clip_skip)}" if clip_skip != 0 else ""
        custom_str = f", {custom}" if custom else ""
        model_hash_str = f", Model hash: {add_model_hash}" if add_model_hash else ""
        hashes_str = f", Hashes: {json.dumps(hashes, separators=(',', ':'))}" if hashes else ""

        a111_params = (
            f"{positive_a111_params}{negative_a111_params}\n"
            f"Steps: {steps}, Sampler: {civitai_sampler_name}, CFG scale: {cfg}, Seed: {seed_value}, "
            f"Size: {width}x{height}{clip_skip_str}{custom_str}{model_hash_str}, Model: {basemodelname}{hashes_str}, Version: ComfyUI"
        )

        if download_civitai_data and civitai_resources:
            a111_params += f", Civitai resources: {json.dumps(civitai_resources, separators=(',', ':'))}"

        all_resources = { modelname: ( ckpt_path, None, modelhash ) } | loras | embeddings | manual_entries
        hash_parts = []
        for name, (_, weight, hash_value) in (all_resources.items() if isinstance(all_resources, dict) else all_resources):
            if name:
                filename = name.split(':')[-1]
                name_without_ext, ext = os.path.splitext(filename)
                supported_extensions = folder_paths.supported_pt_extensions | {".gguf"}
                clean_name = name_without_ext if ext.lower() in supported_extensions else filename
                name_part = f"{clean_name}:"
            else:
                name_part = ""
            if not hash_value: continue
            weight_part = f":{weight}" if weight is not None and download_civitai_data else ""
            hash_parts.append(f"{name_part}{hash_value}{weight_part}")

        final_hashes = ",".join(hash_parts)
        return Metadata(modelname, positive, negative, width, height, seed_value, steps, cfg, sampler_name, scheduler_name, denoise, clip_skip, custom, additional_hashes, ckpt_path, a111_params, final_hashes)

    @staticmethod
    def get_unique_filename(output_path: str, filename_prefix: str, extension: str, batch_size: int = 1, batch_index: int = 0) -> str:
        if not os.path.exists(output_path):
             return f"{filename_prefix}_{batch_index:02d}"

        existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith(extension)]
        if batch_size == 1 and not existing_files:
            return f"{filename_prefix}"

        suffixes: list[int] = []
        for f in existing_files:
            name, _ = os.path.splitext(f)
            parts = name.split('_')
            if parts[-1].isdigit():
                suffixes.append(int(parts[-1]))

        base_suffix = (max(suffixes) + 1) if suffixes else 1
        return f"{filename_prefix}_{base_suffix + batch_index:02d}"


    @staticmethod
    def save_images(
        images: list,
        filename_pattern: str,
        extension: str,
        path: str,
        quality_jpeg_or_webp: int,
        lossless_webp: bool,
        optimize_png: bool,
        prompt: dict[str, Any] | None,
        extra_pnginfo: dict[str, Any] | None,
        save_workflow_as_json: bool,
        embed_workflow: bool,
        counter: int,
        time_format: str,
        metadata: Metadata
    ) -> list[str]:
        filename_prefix = ImageSaverLogic.make_filename(filename_pattern, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip, metadata.custom)
        
        # Apply placeholders to path
        path = ImageSaverLogic.replace_placeholders(path, metadata.width, metadata.height, metadata.seed, metadata.modelname, counter, time_format, metadata.sampler_name, metadata.steps, metadata.cfg, metadata.scheduler_name, metadata.denoise, metadata.clip_skip, metadata.custom)
        
        output_path = os.path.join(folder_paths.output_directory, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                # print(f'UmeAiRT-Image-Saver: Creating directory `{output_path.strip()}`')
                log_node(f'ImageSaver Creating directory `{output_path.strip()}`', color="CYAN")
                os.makedirs(output_path, exist_ok=True)


        result_paths: list[str] = list()
        num_images = len(images)
        for idx, image in enumerate(images):
            # Convert Tensor to PIL
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Simple check for get_unique_filename if it crashes? No, it looks safe.
            current_filename_prefix = ImageSaverLogic.get_unique_filename(output_path, filename_prefix, extension, batch_size=num_images, batch_index=idx)
            final_filename = f"{current_filename_prefix}.{extension}"
            filepath = os.path.join(output_path, final_filename)

            save_image(img, filepath, extension, quality_jpeg_or_webp, lossless_webp, optimize_png, metadata.a111_params, prompt, extra_pnginfo, embed_workflow)

            if save_workflow_as_json:
                save_json(extra_pnginfo, os.path.join(output_path, current_filename_prefix))

            result_paths.append(final_filename)
        return result_paths
