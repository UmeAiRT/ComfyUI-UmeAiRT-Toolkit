
class UmeAiRT_WirelessUltimateUpscale_Advanced(UmeAiRT_WirelessUltimateUpscale_Base):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True}),
                "model": (folder_paths.get_filename_list("upscale_models"),),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05, "display": "slider"}),
                
                # Advanced Settings
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "clean_prompt": ("BOOLEAN", {"default": True, "label_on": "Enable Clean Prompt", "label_off": "Use Global Prompt"}),
                
                "mode_type": (["Linear", "Chess", "None"], {"default": "Linear"}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
                
                "seam_fix_mode": (["None", "Band Pass", "Half Tile", "Half Tile + Intersections"], {"default": "None"}),
                "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": 128, "step": 8}),
                
                "force_uniform_tiles": ("BOOLEAN", {"default": True}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale_advanced"
    CATEGORY = "UmeAiRT/Wireless/Post-Process"

    def upscale_advanced(self, image, enabled, model, upscale_by, denoise, clean_prompt, 
                         mode_type, tile_width, tile_height, mask_blur, tile_padding,
                         seam_fix_mode, seam_fix_denoise, seam_fix_width, seam_fix_mask_blur, seam_fix_padding,
                         force_uniform_tiles, tiled_decode):
        
        if not enabled:
            return (image,)

        # Load Upscale Model
        try:
             from comfy_extras.nodes_upscale_model import UpscaleModelLoader
             upscale_model = UpscaleModelLoader().load_model(model)[0]
        except ImportError:
             raise ImportError("UmeAiRT: Could not import UpscaleModelLoader.")
             
        usdu_node = self.get_usdu_node()
        model, vae, clip, pos_text, neg_text, seed, gen_steps, sampler_name, scheduler, _, _, wireless_cfg, wireless_denoise = self.fetch_wireless_common()
        
        # Clean Prompt
        target_pos_text = "" if clean_prompt else pos_text
        positive, negative = self.encode_prompts(clip, target_pos_text, neg_text)
        
        # Logic
        steps = math.ceil(gen_steps / 4)
        cfg = 1.0 # Force 1.0

        return usdu_node.upscale(
            image=image, model=model, positive=positive, negative=negative, vae=vae,
            upscale_by=upscale_by, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            upscale_model=upscale_model, mode_type=mode_type,
            tile_width=tile_width, tile_height=tile_height, mask_blur=mask_blur, tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode, seam_fix_denoise=seam_fix_denoise,
            seam_fix_mask_blur=seam_fix_mask_blur, seam_fix_width=seam_fix_width, seam_fix_padding=seam_fix_padding,
            force_uniform_tiles=force_uniform_tiles, tiled_decode=tiled_decode,
            suppress_preview=True,
        )
