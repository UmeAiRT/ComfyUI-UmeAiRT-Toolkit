"""
UmeAiRT Toolkit - Node Registration
-----------------------------------
Registers all UmeAiRT nodes and assigns their display names.
Also handles optional resource registration (like 'bbox' folder).
"""

from .nodes import (
    UmeAiRT_Guidance_Input, 
    UmeAiRT_Guidance_Output, 
    UmeAiRT_ImageSize_Input, 
    UmeAiRT_ImageSize_Output,
    UmeAiRT_FPS_Input,
    UmeAiRT_FPS_Output,
    UmeAiRT_Steps_Input,
    UmeAiRT_Steps_Output,
    UmeAiRT_Denoise_Input,
    UmeAiRT_Denoise_Output,
    UmeAiRT_Seed_Input,
    UmeAiRT_Seed_Output,
    UmeAiRT_Scheduler_Input,
    UmeAiRT_Scheduler_Output,
    UmeAiRT_Sampler_Input,
    UmeAiRT_Sampler_Output,
    UmeAiRT_SamplerScheduler_Input,
    UmeAiRT_Positive_Input,
    UmeAiRT_Positive_Output,
    UmeAiRT_Negative_Input,
    UmeAiRT_Negative_Output,
    UmeAiRT_Model_Input,
    UmeAiRT_Model_Output,
    UmeAiRT_VAE_Input,
    UmeAiRT_VAE_Output,
    UmeAiRT_CLIP_Input,
    UmeAiRT_CLIP_Output,
    UmeAiRT_Latent_Input,
    UmeAiRT_Latent_Output,
    UmeAiRT_WirelessKSampler,
    UmeAiRT_Wireless_Debug,
    UmeAiRT_MultiLoraLoader,
    UmeAiRT_WirelessUltimateUpscale,
    UmeAiRT_WirelessUltimateUpscale_Advanced,
    UmeAiRT_WirelessFaceDetailer_Advanced, UmeAiRT_WirelessFaceDetailer_Simple,
    UmeAiRT_BboxDetectorLoader, UmeAiRT_WirelessImageSaver, UmeAiRT_WirelessCheckpointLoader,
    UmeAiRT_WirelessImageLoader, UmeAiRT_SourceImage_Output,
    UmeAiRT_Label
)

# Register internal 'bbox' folder for standalone usage
try:
    import folder_paths
    import os
    models_dir = folder_paths.models_dir
    # Try standard 'bbox' or 'ultralytics/bbox'
    bbox_path = os.path.join(models_dir, "bbox")
    folder_paths.add_model_folder_path("bbox", bbox_path)
    # Also support 'ultralytics/bbox' if users put it there (common standard)
    folder_paths.add_model_folder_path("bbox", os.path.join(models_dir, "ultralytics", "bbox"))
except:
    pass

NODE_CLASS_MAPPINGS = {
    "UmeAiRT_Guidance_Input": UmeAiRT_Guidance_Input,
    "UmeAiRT_Guidance_Output": UmeAiRT_Guidance_Output,
    "UmeAiRT_ImageSize_Input": UmeAiRT_ImageSize_Input,
    "UmeAiRT_ImageSize_Output": UmeAiRT_ImageSize_Output,
    "UmeAiRT_FPS_Input": UmeAiRT_FPS_Input,
    "UmeAiRT_FPS_Output": UmeAiRT_FPS_Output,
    "UmeAiRT_Steps_Input": UmeAiRT_Steps_Input,
    "UmeAiRT_Steps_Output": UmeAiRT_Steps_Output,
    "UmeAiRT_Denoise_Input": UmeAiRT_Denoise_Input,
    "UmeAiRT_Denoise_Output": UmeAiRT_Denoise_Output,
    "UmeAiRT_Seed_Input": UmeAiRT_Seed_Input,
    "UmeAiRT_Seed_Output": UmeAiRT_Seed_Output,
    "UmeAiRT_Scheduler_Input": UmeAiRT_Scheduler_Input,
    "UmeAiRT_Scheduler_Output": UmeAiRT_Scheduler_Output,
    "UmeAiRT_Sampler_Input": UmeAiRT_Sampler_Input,
    "UmeAiRT_Sampler_Output": UmeAiRT_Sampler_Output,
    "UmeAiRT_SamplerScheduler_Input": UmeAiRT_SamplerScheduler_Input,
    "UmeAiRT_Positive_Input": UmeAiRT_Positive_Input,
    "UmeAiRT_Positive_Output": UmeAiRT_Positive_Output,
    "UmeAiRT_Negative_Input": UmeAiRT_Negative_Input,
    "UmeAiRT_Negative_Output": UmeAiRT_Negative_Output,
    "UmeAiRT_WirelessCheckpointLoader": UmeAiRT_WirelessCheckpointLoader,
    "UmeAiRT_WirelessImageLoader": UmeAiRT_WirelessImageLoader,
    "UmeAiRT_SourceImage_Output": UmeAiRT_SourceImage_Output,
    "UmeAiRT_Label": UmeAiRT_Label,
    "UmeAiRT_Model_Input": UmeAiRT_Model_Input,
    "UmeAiRT_Model_Output": UmeAiRT_Model_Output,
    "UmeAiRT_VAE_Input": UmeAiRT_VAE_Input,
    "UmeAiRT_VAE_Output": UmeAiRT_VAE_Output,
    "UmeAiRT_CLIP_Input": UmeAiRT_CLIP_Input,
    "UmeAiRT_CLIP_Output": UmeAiRT_CLIP_Output,
    "UmeAiRT_Latent_Input": UmeAiRT_Latent_Input,
    "UmeAiRT_Latent_Output": UmeAiRT_Latent_Output,
    "UmeAiRT_WirelessKSampler": UmeAiRT_WirelessKSampler,
    "UmeAiRT_Wireless_Debug": UmeAiRT_Wireless_Debug,
    "UmeAiRT_MultiLoraLoader": UmeAiRT_MultiLoraLoader,
    "UmeAiRT_WirelessUltimateUpscale": UmeAiRT_WirelessUltimateUpscale,
    "UmeAiRT_WirelessUltimateUpscale_Advanced": UmeAiRT_WirelessUltimateUpscale_Advanced,
    "UmeAiRT_WirelessFaceDetailer_Advanced": UmeAiRT_WirelessFaceDetailer_Advanced,
    "UmeAiRT_WirelessFaceDetailer_Simple": UmeAiRT_WirelessFaceDetailer_Simple,
    "UmeAiRT_BboxDetectorLoader": UmeAiRT_BboxDetectorLoader,
    "UmeAiRT_WirelessImageSaver": UmeAiRT_WirelessImageSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmeAiRT_Guidance_Input": "Guidance Input",
    "UmeAiRT_Guidance_Output": "Guidance Output",
    "UmeAiRT_ImageSize_Input": "Image Size Input",
    "UmeAiRT_ImageSize_Output": "Image Size Output",
    "UmeAiRT_FPS_Input": "FPS Input",
    "UmeAiRT_FPS_Output": "FPS Output",
    "UmeAiRT_Steps_Input": "Steps Input",
    "UmeAiRT_Steps_Output": "Steps Output",
    "UmeAiRT_Denoise_Input": "Denoise Input",
    "UmeAiRT_Denoise_Output": "Denoise Output",
    "UmeAiRT_Seed_Input": "Seed Input",
    "UmeAiRT_Seed_Output": "Seed Output",
    "UmeAiRT_Scheduler_Input": "Scheduler Input",
    "UmeAiRT_Scheduler_Output": "Scheduler Output",
    "UmeAiRT_Sampler_Input": "Sampler Input",
    "UmeAiRT_Sampler_Output": "Sampler Output",
    "UmeAiRT_SamplerScheduler_Input": "Sampler & Scheduler Input",
    "UmeAiRT_Positive_Input": "Positive Prompt Input",
    "UmeAiRT_Positive_Output": "Positive Prompt Output",
    "UmeAiRT_Negative_Input": "Negative Prompt Input",
    "UmeAiRT_Negative_Output": "Negative Prompt Output",
    "UmeAiRT_Model_Input": "Model Input",
    "UmeAiRT_Model_Output": "Model Output",
    "UmeAiRT_VAE_Input": "VAE Input",
    "UmeAiRT_VAE_Output": "VAE Output",
    "UmeAiRT_CLIP_Input": "CLIP Input",
    "UmeAiRT_CLIP_Output": "CLIP Output",
    "UmeAiRT_Latent_Input": "Latent Input",
    "UmeAiRT_Latent_Output": "Latent Output",
    "UmeAiRT_WirelessKSampler": "Wireless KSampler",
    "UmeAiRT_Wireless_Debug": "Wireless Debug",
    "UmeAiRT_MultiLoraLoader": "Multi-LoRA Loader",
    "UmeAiRT_WirelessUltimateUpscale": "Wireless UltimateSDUpscale",
    "UmeAiRT_WirelessUltimateUpscale_Advanced": "Wireless UltimateSDUpscale (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Advanced": "Wireless FaceDetailer (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Simple": "Wireless FaceDetailer",
    "UmeAiRT_BboxDetectorLoader": "BBOX Detector Loader",
    "UmeAiRT_WirelessImageSaver": "Wireless Image Saver",
    "UmeAiRT_WirelessCheckpointLoader": "Wireless Checkpoint Loader",
    "UmeAiRT_WirelessImageLoader": "Wireless Image Loader",
    "UmeAiRT_SourceImage_Output": "Wireless Source Image",
    "UmeAiRT_Label": "Label"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
