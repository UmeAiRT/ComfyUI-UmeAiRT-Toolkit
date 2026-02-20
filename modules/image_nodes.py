import torch
import numpy as np
import os
import folder_paths
import comfy.utils
import nodes as comfy_nodes
import re
from .common import (
    UME_SHARED_STATE, 
    KEY_SOURCE_IMAGE, 
    KEY_SOURCE_MASK, 
    KEY_IMAGESIZE, 
    KEY_DENOISE, 
    KEY_MODEL_NAME, 
    KEY_SEED,
    KEY_POSITIVE,
    KEY_NEGATIVE,
    KEY_STEPS,
    KEY_CFG,
    KEY_SCHEDULER,
    KEY_SAMPLER,
    KEY_LORAS,
    resize_tensor, 
    log_node
)
from .logger import logger
from .image_saver_core.logic import ImageSaverLogic

class UmeAiRT_WirelessImageLoader(comfy_nodes.LoadImage):
    """
    Advanced Wireless Image Loader.
    Loads an image and updates the global Wireless State (Source Image & Mask).
    Supports optional Resizing and Mode selection (Inpaint vs Img2Img).
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files.sort()
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "resize": ("BOOLEAN", {"default": False, "label_on": "Resize to Global", "label_off": "Keep Original"}),
                "mode": (["Inpaint", "Img2Img"], {"default": "Inpaint", "tooltip": "Inpaint: Use Mask. Img2Img: Ignore Mask."}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image_wireless"
    CATEGORY = "UmeAiRT/Wireless/Loaders"

    def load_image_wireless(self, image, resize, mode):
        # 1. Load Image (Parent Method)
        out = super().load_image(image)
        img = out[0]
        mask = out[1]

        # 2. Resize Logic
        if resize:
            size_dict = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
            target_w = int(size_dict.get("width", 1024))
            target_h = int(size_dict.get("height", 1024))
            
            # Resize Image
            img = resize_tensor(img, target_h, target_w, interp_mode="bilinear", is_mask=False)
            
            # Resize Mask
            if mask is not None:
                mask = resize_tensor(mask, target_h, target_w, interp_mode="nearest", is_mask=True)

        # 3. Update Wireless State
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = img
        
        # Mode Handling
        if mode == "Inpaint":
            UME_SHARED_STATE[KEY_SOURCE_MASK] = mask
        else:
            # Img2Img: Hide mask from wireless state
            UME_SHARED_STATE[KEY_SOURCE_MASK] = None
            
        return (img, mask)

class UmeAiRT_SourceImage_Output:
    """
    Outputs the currently stored Wireless Source Image and Mask.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("source_image", "source_mask")
    FUNCTION = "get_source"
    CATEGORY = "UmeAiRT/Wireless/Accessors"

    def get_source(self):
        img = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)
        
        if img is None:
            # Return dummy black image if missing
            img = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
        if mask is None:
             # Return dummy zero mask
             mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
             
        return (img, mask)

class UmeAiRT_WirelessImageProcess:
    """
    Central node for Wireless Image Editing (Inpaint, Outpaint, Img2Img, Txt2Img).
    Handles Resizing, Padding (Outpaint), Mask Blurring (Inpaint), and Denoise.
    Updates Wireless Global State.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "mode": (["img2img", "inpaint", "outpaint", "txt2img"], {"default": "img2img"}),
            },
            "optional": {
                "image": ("IMAGE",), # Optional input overrides wireless source
                "mask": ("MASK",),   # Optional input overrides wireless mask
                "resize": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                "mask_blur": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                # Outpaint Params
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process_image"
    CATEGORY = "UmeAiRT/Wireless/Pre-Process"

    def process_image(self, denoise=1.0, mode="img2img", image=None, mask=None, resize=False, mask_blur=0, 
                      padding_left=0, padding_top=0, padding_right=0, padding_bottom=0):
        
        if mode == "txt2img":
             log_node("Wireless Process: Txt2Img Mode (Forcing Denoise=1.0, Ignoring Mask).", color="YELLOW")
             # Force Denoise
             UME_SHARED_STATE[KEY_DENOISE] = 1.0
             # Hide Mask
             UME_SHARED_STATE[KEY_SOURCE_MASK] = None
             
             if image is not None:
                  UME_SHARED_STATE[KEY_SOURCE_IMAGE] = image
             
             return (image, None)

        # 1. Fetch Input (Priority: Wired > Wireless)
        if image is None:
            image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        if mask is None:
            mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)
        
        if image is None:
            UME_SHARED_STATE[KEY_DENOISE] = denoise
            return (None, None)

        B, H, W, C = image.shape
        
        # Determine Target Size (Wireless State)
        target_w = W
        target_h = H
        if resize:
             size = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
             target_w = int(size.get("width", 1024))
             target_h = int(size.get("height", 1024))

        # 2. Process based on Mode
        final_image = image
        final_mask = mask
        
        # RESIZE (Pre-Outpaint)
        if resize:
             final_image = resize_tensor(final_image, target_h, target_w, interp_mode="bilinear", is_mask=False)
             if final_mask is not None:
                 final_mask = resize_tensor(final_mask, target_h, target_w, interp_mode="nearest", is_mask=True)
             
             # Update dims
             B, H, W, C = final_image.shape

        # OUTPAINT
        if mode == "outpaint":
             pad_l, pad_t, pad_r, pad_b = padding_left, padding_top, padding_right, padding_bottom
             
             if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
                 # Pad Image (Replicate to stretch edges)
                 img_p = final_image.permute(0, 3, 1, 2)
                 img_padded = torch.nn.functional.pad(img_p, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
                 final_image = img_padded.permute(0, 2, 3, 1)
                 
                 new_h = H + pad_t + pad_b
                 new_w = W + pad_l + pad_r
                 
                 # Create Mask logic for Outpaint
                 
                 # Start with Zeros (Keep everything)
                 new_mask = torch.zeros((B, new_h, new_w), dtype=torch.float32, device=final_image.device)
                 
                 # If original mask existed, pad it
                 if final_mask is not None:
                     # final_mask is [B, H, W]
                     # Check dims
                     if len(final_mask.shape) == 2: m_in = final_mask.unsqueeze(0)
                     else: m_in = final_mask
                     
                     # Pad original mask with 0 (preserve existing mask content)
                     m_padded = torch.nn.functional.pad(m_in, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                     if len(final_mask.shape) == 2: new_mask = m_padded.squeeze(0)
                     else: new_mask = m_padded

                 # Set Padded Areas to 1.0 (Inpaint)
                 # We add a slight overlap (8 px) into the original image so the AI can blend the edge seamlessly
                 overlap = 8
                 if pad_t > 0: new_mask[:, :pad_t + overlap, :] = 1.0
                 if pad_b > 0: new_mask[:, -(pad_b + overlap):, :] = 1.0
                 if pad_l > 0: new_mask[:, :, :pad_l + overlap] = 1.0
                 if pad_r > 0: new_mask[:, :, -(pad_r + overlap):] = 1.0
                 
                 # Feathering logic
                 feathering = 40
                 if feathering > 0:
                      import torchvision.transforms.functional as TF
                      k = feathering
                      if k % 2 == 0: k += 1
                      sig = float(k) / 3.0
                      
                      if len(new_mask.shape) == 2: m_b = new_mask.unsqueeze(0).unsqueeze(0)
                      else: m_b = new_mask.unsqueeze(1)
                      
                      m_b = TF.gaussian_blur(m_b, kernel_size=k, sigma=sig)
                      
                      if len(new_mask.shape) == 2: new_mask = m_b.squeeze(0).squeeze(0)
                      else: new_mask = m_b.squeeze(1)
                 
                 final_mask = new_mask

        # INPAINT / COMPOSITE BLUR logic preparation
        if mask_blur > 0 and final_mask is not None:
             if len(final_mask.shape) == 2: m = final_mask.unsqueeze(0).unsqueeze(0)
             elif len(final_mask.shape) == 3: m = final_mask.unsqueeze(1)
             else: m = final_mask
             
             import torchvision.transforms.functional as TF
             k = mask_blur
             if k % 2 == 0: k += 1
             m = TF.gaussian_blur(m, kernel_size=k)
             
             if len(final_mask.shape) == 2: final_mask = m.squeeze(0).squeeze(0)
             elif len(final_mask.shape) == 3: final_mask = m.squeeze(1)

        # MODE HANDLING (State Update)
        state_mask = final_mask
        if mode == "img2img":
            state_mask = None # Hide mask from state
            log_node("Wireless Process: Img2Img Mode (State Mask Hidden).", color="YELLOW")
        
        # 6. Update State
        UME_SHARED_STATE[KEY_SOURCE_IMAGE] = final_image
        UME_SHARED_STATE[KEY_SOURCE_MASK] = state_mask
        UME_SHARED_STATE[KEY_DENOISE] = denoise

        return (final_image, final_mask)


class UmeAiRT_WirelessInpaintComposite:
    """
    Composites the generated image onto the original source image using the Wireless Mask.
    Useful for seamless Inpainting.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "generated_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite_image",)
    FUNCTION = "composite"
    CATEGORY = "UmeAiRT/Wireless/Post-Processing"

    def composite(self, generated_image):
        # Fetch Source from Wireless
        source_image = UME_SHARED_STATE.get(KEY_SOURCE_IMAGE)
        source_mask = UME_SHARED_STATE.get(KEY_SOURCE_MASK)

        if source_image is None or source_mask is None:
            log_node("Wireless Composite: Missing source image or mask! Returning generated image.", color="RED")
            return (generated_image,)

        # Dimensions
        gB, gH, gW, gC = generated_image.shape
        sB, sH, sW, sC = source_image.shape
        
        # Helper to resize source to generated
        # Why generated? Because logic might have upscaled/down-scaled the latent.
        # We usually want to match the generated resolution.
        
        # Resize Source Image
        source_resized = source_image
        if sH != gH or sW != gW:
            source_resized = resize_tensor(source_image, gH, gW, interp_mode="bilinear", is_mask=False)
            
        # Resize Mask
        mask_resized = source_mask
        # Mask dim check
        if len(mask_resized.shape) == 2:
            mH, mW = mask_resized.shape
        elif len(mask_resized.shape) == 3:
             # B, H, W (Comfy mask is usually B,H,W or H,W)
             mB, mH, mW = mask_resized.shape
        else:
            mH, mW = gH, gW # Fallback

        if mH != gH or mW != gW:
             mask_resized = resize_tensor(source_mask, gH, gW, interp_mode="bilinear", is_mask=True) # Bilinear for soft edges

        # Composite Algorithm
        # Image = Source * (1-Mask) + Generated * Mask
        
        # Prepare Mask for broadcasting
        m = mask_resized
        if len(m.shape) == 2:
            m = m.unsqueeze(0).unsqueeze(-1) # 1, H, W, 1
        elif len(m.shape) == 3:
            m = m.unsqueeze(-1) # B, H, W, 1
            
        # Ensure Batch Match
        if m.shape[0] < gB:
            m = m.repeat(gB, 1, 1, 1)
        if source_resized.shape[0] < gB:
            source_resized = source_resized.repeat(gB, 1, 1, 1)

        # In Comfy, Mask: 1=Inpaint(Change), 0=Keep
        # So we want to keep Source where Mask=0
        # Result = Source * (1-Mask) + Generated * Mask
        
        composite = source_resized * (1.0 - m) + generated_image * m
        
        return (composite,)


class UmeAiRT_WirelessImageSaver:
    """
    Saves the image with filename based on Wireless settings (Model Name, Seed, Date/Time).
    Includes simplified metadata.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "%date%_%time%_%model%_%seed%", "multiline": False}),
            },
            "hidden": {
                 "prompt": "PROMPT",
                 "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_images"
    CATEGORY = "UmeAiRT/Wireless/IO"
    
    def save_images(self, images, filename, prompt=None, extra_pnginfo=None):
        # 1. Resolve Path and Filename (Standardize splitting)
        full_pattern = filename.replace("\\", "/")
        if "/" in full_pattern:
             path, filename = full_pattern.rsplit("/", 1)
        else:
             path = ""
             filename = full_pattern
        
        # Sanitize Filename (Manager Request)
        # Remove invalid characters: < > : " / \ | ? *
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        
        # Hardcoded Defaults for Simple Mode
        extension = "png"
        lossless_webp = True
        quality_jpeg_or_webp = 100
        optimize_png = False
        embed_workflow = True
        save_workflow_as_json = False
        
        # 2. Fetch shared state
        width = UME_SHARED_STATE.get(KEY_IMAGESIZE, {}).get("width", 512)
        height = UME_SHARED_STATE.get(KEY_IMAGESIZE, {}).get("height", 512)
        modelname = UME_SHARED_STATE.get(KEY_MODEL_NAME, "UmeAiRT_Wireless_Model")
        
        # Process LoRAs for metadata hashes
        additional_hashes = ""
        loras = UME_SHARED_STATE.get(KEY_LORAS, [])
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

        # 3. Build Metadata Object
        try:
            metadata_obj = ImageSaverLogic.make_metadata(
                modelname=modelname,
                positive=UME_SHARED_STATE.get(KEY_POSITIVE, ""),
                negative=UME_SHARED_STATE.get(KEY_NEGATIVE, ""),
                width=int(width),
                height=int(height),
                seed_value=int(UME_SHARED_STATE.get(KEY_SEED, 0)),
                steps=int(UME_SHARED_STATE.get(KEY_STEPS, 20)),
                cfg=float(UME_SHARED_STATE.get(KEY_CFG, 8.0)),
                sampler_name=UME_SHARED_STATE.get(KEY_SAMPLER, "euler"),
                scheduler_name=UME_SHARED_STATE.get(KEY_SCHEDULER, "normal"),
                denoise=float(UME_SHARED_STATE.get(KEY_DENOISE, 1.0)),
                clip_skip=0,
                custom="UmeAiRT Wireless",
                additional_hashes=additional_hashes,
                download_civitai_data=False,
                easy_remix=True
            )
        except Exception as e:
            log_node(f"Metadata Creation Failed: {e}", color="RED")
            raise e

        # 4. Save
        time_format = "%Y-%m-%d-%H%M%S"
        
        # Resolve Path Placeholders (for UI return Consistency)
        resolved_path = ImageSaverLogic.replace_placeholders(
            path, 
            metadata_obj.width, metadata_obj.height, metadata_obj.seed, metadata_obj.modelname, 
            0, time_format, 
            metadata_obj.sampler_name, metadata_obj.steps, metadata_obj.cfg, metadata_obj.scheduler_name, 
            metadata_obj.denoise, metadata_obj.clip_skip, metadata_obj.custom
        )
        
        try:
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
                counter=0,
                time_format=time_format,
                metadata=metadata_obj
            )
            
            if len(result_filenames) == 1:
                log_node(f"Image Saver: Saved -> {resolved_path}/{result_filenames[0]}", color="GREEN")
            else:
                log_node(f"Image Saver: Saved {len(result_filenames)} images -> {resolved_path} (e.g. {result_filenames[0]})", color="GREEN")
            
            # 5. Format Output UI
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


