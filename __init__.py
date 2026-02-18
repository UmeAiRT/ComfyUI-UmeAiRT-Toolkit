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
    UmeAiRT_WirelessSeedVR2Upscale, UmeAiRT_WirelessSeedVR2Upscale_Advanced,
    UmeAiRT_WirelessFaceDetailer_Advanced, UmeAiRT_WirelessFaceDetailer_Simple,
    UmeAiRT_BboxDetectorLoader, UmeAiRT_WirelessImageSaver, UmeAiRT_WirelessCheckpointLoader,
    UmeAiRT_WirelessImageLoader, UmeAiRT_SourceImage_Output, UmeAiRT_WirelessInpaintComposite,
    UmeAiRT_Label, UmeAiRT_WirelessImageProcess,
    UmeAiRT_GenerationSettings, UmeAiRT_FilesSettings_Checkpoint, UmeAiRT_FilesSettings_Checkpoint_Advanced, UmeAiRT_FilesSettings_FLUX, UmeAiRT_BlockSampler, UmeAiRT_PromptBlock,
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3, UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    UmeAiRT_BlockUltimateSDUpscale, UmeAiRT_BlockFaceDetailer, UmeAiRT_BlockImageLoader, UmeAiRT_BlockImageLoader_Advanced, UmeAiRT_BlockImageProcess,
    UmeAiRT_Unpack_ImageBundle, UmeAiRT_Unpack_FilesBundle, UmeAiRT_Unpack_SettingsBundle, UmeAiRT_Unpack_PromptsBundle,
    UmeAiRT_ControlNetImageApply_Simple, UmeAiRT_ControlNetImageApply_Advanced, UmeAiRT_ControlNetImageProcess,
    
    # Tools
    UmeAiRT_Bundle_Downloader,
    UmeAiRT_Log_Viewer,
)
from .optimization_utils import check_optimizations

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
    # Inputs/Outputs (Raw)
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
    "UmeAiRT_Model_Input": UmeAiRT_Model_Input,
    "UmeAiRT_Model_Output": UmeAiRT_Model_Output,
    "UmeAiRT_VAE_Input": UmeAiRT_VAE_Input,
    "UmeAiRT_VAE_Output": UmeAiRT_VAE_Output,
    "UmeAiRT_CLIP_Input": UmeAiRT_CLIP_Input,
    "UmeAiRT_CLIP_Output": UmeAiRT_CLIP_Output,
    "UmeAiRT_Latent_Input": UmeAiRT_Latent_Input,
    "UmeAiRT_Latent_Output": UmeAiRT_Latent_Output,
    
    # Wireless (Getters/Processors)
    "UmeAiRT_WirelessKSampler": UmeAiRT_WirelessKSampler,
    "UmeAiRT_Wireless_Debug": UmeAiRT_Wireless_Debug,
    "UmeAiRT_MultiLoraLoader": UmeAiRT_MultiLoraLoader,
    "UmeAiRT_WirelessUltimateUpscale": UmeAiRT_WirelessUltimateUpscale,
    "UmeAiRT_WirelessUltimateUpscale_Advanced": UmeAiRT_WirelessUltimateUpscale_Advanced,
    "UmeAiRT_WirelessSeedVR2Upscale": UmeAiRT_WirelessSeedVR2Upscale,
    "UmeAiRT_WirelessSeedVR2Upscale_Advanced": UmeAiRT_WirelessSeedVR2Upscale_Advanced,
    "UmeAiRT_WirelessFaceDetailer_Advanced": UmeAiRT_WirelessFaceDetailer_Advanced,
    "UmeAiRT_WirelessFaceDetailer_Simple": UmeAiRT_WirelessFaceDetailer_Simple,
    "UmeAiRT_BboxDetectorLoader": UmeAiRT_BboxDetectorLoader,
    "UmeAiRT_WirelessImageSaver": UmeAiRT_WirelessImageSaver,
    "UmeAiRT_WirelessCheckpointLoader": UmeAiRT_WirelessCheckpointLoader,
    "UmeAiRT_WirelessImageLoader": UmeAiRT_WirelessImageLoader,
    "UmeAiRT_SourceImage_Output": UmeAiRT_SourceImage_Output,
    "UmeAiRT_WirelessInpaintComposite": UmeAiRT_WirelessInpaintComposite,
    "UmeAiRT_Label": UmeAiRT_Label,
    "UmeAiRT_WirelessImageProcess": UmeAiRT_WirelessImageProcess,

    # Block Nodes
    "UmeAiRT_GenerationSettings": UmeAiRT_GenerationSettings,
    "UmeAiRT_FilesSettings_Checkpoint": UmeAiRT_FilesSettings_Checkpoint,
    "UmeAiRT_FilesSettings_Checkpoint_Advanced": UmeAiRT_FilesSettings_Checkpoint_Advanced,
    "UmeAiRT_FilesSettings_FLUX": UmeAiRT_FilesSettings_FLUX,
    "UmeAiRT_LoraBlock_1": UmeAiRT_LoraBlock_1,
    "UmeAiRT_LoraBlock_3": UmeAiRT_LoraBlock_3,
    "UmeAiRT_LoraBlock_5": UmeAiRT_LoraBlock_5,
    "UmeAiRT_LoraBlock_10": UmeAiRT_LoraBlock_10,
    "UmeAiRT_PromptBlock": UmeAiRT_PromptBlock,
    "UmeAiRT_BlockSampler": UmeAiRT_BlockSampler,
    "UmeAiRT_BlockUltimateSDUpscale": UmeAiRT_BlockUltimateSDUpscale,
    "UmeAiRT_BlockFaceDetailer": UmeAiRT_BlockFaceDetailer,
    "UmeAiRT_BlockImageLoader": UmeAiRT_BlockImageLoader,
    "UmeAiRT_BlockImageLoader_Advanced": UmeAiRT_BlockImageLoader_Advanced,
    "UmeAiRT_BlockImageProcess": UmeAiRT_BlockImageProcess,

    # Unpack Nodes
    "UmeAiRT_Unpack_ImageBundle": UmeAiRT_Unpack_ImageBundle,
    "UmeAiRT_Unpack_FilesBundle": UmeAiRT_Unpack_FilesBundle,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,
    "UmeAiRT_Unpack_PromptsBundle": UmeAiRT_Unpack_PromptsBundle,
    
    # ControlNet
    "UmeAiRT_ControlNetImageApply_Simple": UmeAiRT_ControlNetImageApply_Simple,
    "UmeAiRT_ControlNetImageApply_Advanced": UmeAiRT_ControlNetImageApply_Advanced,
    "UmeAiRT_ControlNetImageProcess": UmeAiRT_ControlNetImageProcess,
    
    # Tools
    "UmeAiRT_Bundle_Downloader": UmeAiRT_Bundle_Downloader,
    "UmeAiRT_Log_Viewer": UmeAiRT_Log_Viewer,
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
    "UmeAiRT_WirelessKSampler": "KSampler (Wireless)",
    "UmeAiRT_Wireless_Debug": "Wireless Debug",
    "UmeAiRT_MultiLoraLoader": "Multi-LoRA Loader",
    "UmeAiRT_WirelessUltimateUpscale": "Wireless UltimateSDUpscale",
    "UmeAiRT_WirelessUltimateUpscale_Advanced": "Wireless UltimateSDUpscale (Advanced)",
    "UmeAiRT_WirelessSeedVR2Upscale": "Wireless SeedVR2 Upscale",
    "UmeAiRT_WirelessSeedVR2Upscale_Advanced": "Wireless SeedVR2 Upscale (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Advanced": "Wireless FaceDetailer (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Simple": "Wireless FaceDetailer",
    "UmeAiRT_BboxDetectorLoader": "BBOX Detector Loader",
    "UmeAiRT_WirelessImageSaver": "Wireless Image Saver",
    "UmeAiRT_WirelessCheckpointLoader": "Wireless Checkpoint Loader",
    "UmeAiRT_WirelessImageLoader": "Wireless Image Loader",
    "UmeAiRT_SourceImage_Output": "Wireless Source Image",
    "UmeAiRT_WirelessInpaintComposite": "Wireless Inpaint Composite",
    "UmeAiRT_Label": "Label",
    "UmeAiRT_WirelessImageProcess": "Wireless Image Process",
    "UmeAiRT_GenerationSettings": "Generation Settings (Block)",
    "UmeAiRT_FilesSettings_Checkpoint": "Model Loader (Block)",
    "UmeAiRT_FilesSettings_Checkpoint_Advanced": "Model Loader - Advanced (Block)",
    "UmeAiRT_FilesSettings_FLUX": "Model Loader - FLUX (Block)",
    "UmeAiRT_PromptBlock": "Prompts (Block)",
    "UmeAiRT_LoraBlock_1": "LoRA 1x (Block)",
    "UmeAiRT_LoraBlock_3": "LoRA 3x (Block)",
    "UmeAiRT_LoraBlock_5": "LoRA 5x (Block)",
    "UmeAiRT_LoraBlock_10": "LoRA 10x (Block)",
    "UmeAiRT_BlockSampler": "Block Sampler",
    "UmeAiRT_BlockUltimateSDUpscale": "UltimateSD Upscale (Block)",
    "UmeAiRT_BlockFaceDetailer": "Face Detailer (Block)",
    "UmeAiRT_BlockImageLoader": "Image Loader (Block)",
    "UmeAiRT_BlockImageLoader_Advanced": "Image Loader - Advanced (Block)",
    "UmeAiRT_BlockImageProcess": "Image Process (Block)",
    "UmeAiRT_Unpack_ImageBundle": "Unpack Image Bundle",
    "UmeAiRT_Unpack_FilesBundle": "Unpack Models Bundle",
    "UmeAiRT_Unpack_SettingsBundle": "Unpack Settings Bundle",
    "UmeAiRT_Unpack_SettingsBundle": "Unpack Settings Bundle",
    "UmeAiRT_Unpack_PromptsBundle": "Unpack Prompts Bundle",
    "UmeAiRT_ControlNetImageApply_Simple": "ControlNet Apply (Simple)",
    "UmeAiRT_ControlNetImageApply_Advanced": "ControlNet Apply (Advanced)",
    "UmeAiRT_ControlNetImageProcess": "ControlNet Process (Unified)",
    
    # Tools
    "UmeAiRT_Bundle_Downloader": "💾 Bundle Model Downloader",
    "UmeAiRT_Log_Viewer": "📜 UmeAiRT Log Viewer",
}

WEB_DIRECTORY = "./web"

# --- STARTUP LOGGING ---
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(convert=True, autoreset=True)
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    RED = Fore.RED
    RESET = Style.RESET_ALL
except ImportError:
    CYAN = GREEN = RED = RESET = ""

print(f"[{CYAN}UmeAiRT-Toolkit{RESET}] 📂 Loading nodes...")
print(f"[{CYAN}UmeAiRT-Toolkit{RESET}] 🧩 Loaded {len(NODE_CLASS_MAPPINGS)} nodes.")
check_optimizations()
print(f"[{CYAN}UmeAiRT-Toolkit{RESET}]{GREEN} ✅ Initialization Complete.{RESET}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
