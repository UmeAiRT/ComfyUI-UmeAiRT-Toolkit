# --- Dynamic Sampler Registration ---
try:
    from .modules.extra_samplers import register_extra_samplers
    register_extra_samplers()
except Exception as e:
    print(f"[UmeAiRT-Toolkit] Failed to register extra samplers: {e}")
# ------------------------------------

from .modules.settings_nodes import (
    UmeAiRT_GlobalSeed,
    UmeAiRT_Resolution,
    UmeAiRT_Prompt,
    UmeAiRT_SpeedMode,
    # Wireless Variables (Inputs/Outputs)
    UmeAiRT_Guidance_Input, UmeAiRT_Guidance_Output,
    UmeAiRT_ImageSize_Input, UmeAiRT_ImageSize_Output,
    UmeAiRT_FPS_Input, UmeAiRT_FPS_Output,
    UmeAiRT_Steps_Input, UmeAiRT_Steps_Output,
    UmeAiRT_Seed_Input, UmeAiRT_Seed_Output,
    UmeAiRT_Denoise_Input, UmeAiRT_Denoise_Output,
    UmeAiRT_Sampler_Input, UmeAiRT_Sampler_Output,
    UmeAiRT_Scheduler_Input, UmeAiRT_Scheduler_Output,
    UmeAiRT_SamplerScheduler_Input,
    UmeAiRT_Positive_Input, UmeAiRT_Positive_Output,
    UmeAiRT_Negative_Input, UmeAiRT_Negative_Output
)
from .modules.model_nodes import (
    UmeAiRT_WirelessModelLoader,
    UmeAiRT_Model_Input, UmeAiRT_Model_Output,
    UmeAiRT_VAE_Input, UmeAiRT_VAE_Output,
    UmeAiRT_CLIP_Input, UmeAiRT_CLIP_Output,
    UmeAiRT_Latent_Input, UmeAiRT_Latent_Output,
    UmeAiRT_WirelessCheckpointLoader,
    UmeAiRT_MultiLoraLoader
)
from .modules.image_nodes import (
    UmeAiRT_WirelessImageLoader,
    UmeAiRT_SourceImage_Output,
    UmeAiRT_WirelessImageProcess,
    UmeAiRT_WirelessInpaintComposite,
    UmeAiRT_WirelessImageSaver
)

# Register 'bbox' folder for FaceDetailer
import folder_paths
import os
try:
    folder_paths.add_model_folder_path("bbox", os.path.join(folder_paths.models_dir, "bbox"))
except Exception:
    pass
# Ensure it exists in folder_names_and_paths just in case add_model_folder_path didn't create the key (older comfy versions)
if "bbox" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["bbox"] = ([os.path.join(folder_paths.models_dir, "bbox")], folder_paths.supported_pt_extensions)

from .modules.logic_nodes import (
    UmeAiRT_WirelessKSampler,
    UmeAiRT_WirelessUltimateUpscale,
    UmeAiRT_WirelessUltimateUpscale_Advanced,
    UmeAiRT_WirelessSeedVR2Upscale,
    UmeAiRT_WirelessSeedVR2Upscale_Advanced,
    UmeAiRT_BboxDetectorLoader,
    UmeAiRT_WirelessFaceDetailer_Advanced,
    UmeAiRT_WirelessFaceDetailer_Simple,
    UmeAiRT_Detailer_Daemon_Simple,
    UmeAiRT_Detailer_Daemon_Advanced
)
from .modules.block_nodes import (
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3, UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    UmeAiRT_ControlNetImageApply_Advanced, UmeAiRT_ControlNetImageApply_Simple, UmeAiRT_ControlNetImageProcess,
    UmeAiRT_GenerationSettings,
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_Checkpoint_Advanced,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_FilesSettings_Fragmented,
    UmeAiRT_FilesSettings_ZIMG,
    UmeAiRT_BlockImageLoader,
 UmeAiRT_BlockImageLoader_Advanced, UmeAiRT_BlockImageProcess,
    UmeAiRT_BlockSampler, UmeAiRT_BlockUltimateSDUpscale, UmeAiRT_BlockFaceDetailer,
    UmeAiRT_BundleLoader
)
from .modules.utils_nodes import (
    UmeAiRT_Label,
    UmeAiRT_Wireless_Debug,
    UmeAiRT_Bundle_Downloader,
    UmeAiRT_Log_Viewer,
    UmeAiRT_Unpack_Settings,
    UmeAiRT_Unpack_Prompt,
    # Restored Unpack Nodes
    UmeAiRT_Faces_Unpack_Node,
    UmeAiRT_Tags_Unpack_Node,
    UmeAiRT_Pipe_Unpack_Node,
    UmeAiRT_Unpack_SettingsBundle,
    UmeAiRT_Unpack_PromptsBundle,
    UmeAiRT_Unpack_ImageBundle,
    UmeAiRT_Unpack_FilesBundle,
    UmeAiRT_Signature,
    UmeAiRT_HealthCheck
)

NODE_CLASS_MAPPINGS = {
    # Settings
    "UmeAiRT_GlobalSeed": UmeAiRT_GlobalSeed,
    "UmeAiRT_Resolution": UmeAiRT_Resolution,
    "UmeAiRT_Prompt": UmeAiRT_Prompt,
    "UmeAiRT_SpeedMode": UmeAiRT_SpeedMode,


    # Loaders (Wireless)
    "UmeAiRT_WirelessModelLoader": UmeAiRT_WirelessModelLoader,

    # Wireless Variables
    "UmeAiRT_Guidance_Input": UmeAiRT_Guidance_Input,
    "UmeAiRT_Guidance_Output": UmeAiRT_Guidance_Output,
    "UmeAiRT_ImageSize_Input": UmeAiRT_ImageSize_Input,
    "UmeAiRT_ImageSize_Output": UmeAiRT_ImageSize_Output,
    "UmeAiRT_FPS_Input": UmeAiRT_FPS_Input,
    "UmeAiRT_FPS_Output": UmeAiRT_FPS_Output,
    "UmeAiRT_Steps_Input": UmeAiRT_Steps_Input,
    "UmeAiRT_Steps_Output": UmeAiRT_Steps_Output,
    "UmeAiRT_Seed_Input": UmeAiRT_Seed_Input,
    "UmeAiRT_Seed_Output": UmeAiRT_Seed_Output,
    "UmeAiRT_Denoise_Input": UmeAiRT_Denoise_Input,
    "UmeAiRT_Denoise_Output": UmeAiRT_Denoise_Output,
    "UmeAiRT_Sampler_Input": UmeAiRT_Sampler_Input,
    "UmeAiRT_Sampler_Output": UmeAiRT_Sampler_Output,
    "UmeAiRT_Scheduler_Input": UmeAiRT_Scheduler_Input,
    "UmeAiRT_Scheduler_Output": UmeAiRT_Scheduler_Output,
    "UmeAiRT_SamplerScheduler_Input": UmeAiRT_SamplerScheduler_Input,
    "UmeAiRT_Positive_Input": UmeAiRT_Positive_Input,
    "UmeAiRT_Positive_Output": UmeAiRT_Positive_Output,
    "UmeAiRT_Negative_Input": UmeAiRT_Negative_Input,
    "UmeAiRT_Negative_Output": UmeAiRT_Negative_Output,
    
    # Model Variables
    "UmeAiRT_Model_Input": UmeAiRT_Model_Input,
    "UmeAiRT_Model_Output": UmeAiRT_Model_Output,
    "UmeAiRT_VAE_Input": UmeAiRT_VAE_Input,
    "UmeAiRT_VAE_Output": UmeAiRT_VAE_Output,
    "UmeAiRT_CLIP_Input": UmeAiRT_CLIP_Input,
    "UmeAiRT_CLIP_Output": UmeAiRT_CLIP_Output,
    "UmeAiRT_Latent_Input": UmeAiRT_Latent_Input,
    "UmeAiRT_Latent_Output": UmeAiRT_Latent_Output,

    # Seed Aliases
    "UmeAiRT_Seed_Node": UmeAiRT_Seed_Input,
    "UmeAiRT_CR_Seed_Node": UmeAiRT_Seed_Input,
 
    # Restored Unpack Nodes
    "UmeAiRT_Faces_Unpack_Node": UmeAiRT_Faces_Unpack_Node,
    "UmeAiRT_Tags_Unpack_Node": UmeAiRT_Tags_Unpack_Node,
    "UmeAiRT_Pipe_Unpack_Node": UmeAiRT_Pipe_Unpack_Node,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,
    "UmeAiRT_Unpack_PromptsBundle": UmeAiRT_Unpack_PromptsBundle,
    "UmeAiRT_Unpack_ImageBundle": UmeAiRT_Unpack_ImageBundle,
    "UmeAiRT_Unpack_FilesBundle": UmeAiRT_Unpack_FilesBundle,

    
    # Files Loaders (Block/Fragmented)
    "UmeAiRT_FilesSettings_Checkpoint": UmeAiRT_FilesSettings_Checkpoint,
    "UmeAiRT_FilesSettings_FLUX": UmeAiRT_FilesSettings_FLUX,
    "UmeAiRT_FilesSettings_Fragmented": UmeAiRT_FilesSettings_Fragmented,
    "UmeAiRT_FilesSettings_ZIMG": UmeAiRT_FilesSettings_ZIMG,

    # Image (Wireless)
    "UmeAiRT_WirelessImageLoader": UmeAiRT_WirelessImageLoader,
    "UmeAiRT_SourceImage_Output": UmeAiRT_SourceImage_Output,
    "UmeAiRT_WirelessImageProcess": UmeAiRT_WirelessImageProcess,
    "UmeAiRT_WirelessInpaintComposite": UmeAiRT_WirelessInpaintComposite,
    "UmeAiRT_WirelessImageSaver": UmeAiRT_WirelessImageSaver,

    # Logic / Samplers
    "UmeAiRT_WirelessCheckpointLoader": UmeAiRT_WirelessCheckpointLoader,
    "UmeAiRT_WirelessKSampler": UmeAiRT_WirelessKSampler,
    "UmeAiRT_WirelessUltimateUpscale": UmeAiRT_WirelessUltimateUpscale,
    "UmeAiRT_WirelessUltimateUpscale_Advanced": UmeAiRT_WirelessUltimateUpscale_Advanced,
    "UmeAiRT_WirelessSeedVR2Upscale": UmeAiRT_WirelessSeedVR2Upscale,
    "UmeAiRT_WirelessSeedVR2Upscale_Advanced": UmeAiRT_WirelessSeedVR2Upscale_Advanced,
    "UmeAiRT_BboxDetectorLoader": UmeAiRT_BboxDetectorLoader,
    "UmeAiRT_WirelessFaceDetailer_Advanced": UmeAiRT_WirelessFaceDetailer_Advanced,
    "UmeAiRT_WirelessFaceDetailer_Simple": UmeAiRT_WirelessFaceDetailer_Simple,
    "UmeAiRT_Detailer_Daemon_Simple": UmeAiRT_Detailer_Daemon_Simple,
    "UmeAiRT_Detailer_Daemon_Advanced": UmeAiRT_Detailer_Daemon_Advanced,

    # Blocks
    "UmeAiRT_LoraBlock_1": UmeAiRT_LoraBlock_1,
    "UmeAiRT_LoraBlock_3": UmeAiRT_LoraBlock_3,
    "UmeAiRT_LoraBlock_5": UmeAiRT_LoraBlock_5,
    "UmeAiRT_LoraBlock_10": UmeAiRT_LoraBlock_10,
    "UmeAiRT_ControlNetImageApply_Advanced": UmeAiRT_ControlNetImageApply_Advanced,
    "UmeAiRT_ControlNetImageApply_Simple": UmeAiRT_ControlNetImageApply_Simple,
    "UmeAiRT_ControlNetImageProcess": UmeAiRT_ControlNetImageProcess,
    "UmeAiRT_GenerationSettings": UmeAiRT_GenerationSettings,
    "UmeAiRT_FilesSettings_Checkpoint_Advanced": UmeAiRT_FilesSettings_Checkpoint_Advanced,
    "UmeAiRT_BlockImageLoader": UmeAiRT_BlockImageLoader,
    "UmeAiRT_BlockImageLoader_Advanced": UmeAiRT_BlockImageLoader_Advanced,
    "UmeAiRT_BlockImageProcess": UmeAiRT_BlockImageProcess,
    "UmeAiRT_BlockSampler": UmeAiRT_BlockSampler,
    "UmeAiRT_BlockUltimateSDUpscale": UmeAiRT_BlockUltimateSDUpscale,
    "UmeAiRT_BlockFaceDetailer": UmeAiRT_BlockFaceDetailer,

    # Utils
    "UmeAiRT_Label": UmeAiRT_Label,
    "UmeAiRT_Signature": UmeAiRT_Signature,
    "UmeAiRT_Wireless_Debug": UmeAiRT_Wireless_Debug,
    "UmeAiRT_Bundle_Downloader": UmeAiRT_Bundle_Downloader,
    "UmeAiRT_BundleLoader": UmeAiRT_BundleLoader,
    "UmeAiRT_Log_Viewer": UmeAiRT_Log_Viewer,
    "UmeAiRT_HealthCheck": UmeAiRT_HealthCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Settings
    "UmeAiRT_GlobalSeed": "Global Seed",
    "UmeAiRT_Resolution": "Resolution",
    "UmeAiRT_Prompt": "Prompt",
    "UmeAiRT_SpeedMode": "Speed Mode",
    "UmeAiRT_Seed_Node": "Seed",
    "UmeAiRT_CR_Seed_Node": "CR Seed",

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
    "UmeAiRT_WirelessUltimateUpscale": "Wireless UltimateSDUpscale",
    "UmeAiRT_WirelessUltimateUpscale_Advanced": "Wireless UltimateSDUpscale (Advanced)",
    "UmeAiRT_WirelessSeedVR2Upscale": "Wireless SeedVR2 Upscale",
    "UmeAiRT_WirelessSeedVR2Upscale_Advanced": "Wireless SeedVR2 Upscale (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Advanced": "Wireless FaceDetailer (Advanced)",
    "UmeAiRT_WirelessFaceDetailer_Simple": "Wireless FaceDetailer",
    "UmeAiRT_BboxDetectorLoader": "BBOX Detector Loader",
    "UmeAiRT_WirelessImageSaver": "Wireless Image Saver",
    "UmeAiRT_WirelessCheckpointLoader": "Wireless Checkpoint Loader",
    "UmeAiRT_WirelessModelLoader": "Wireless Model Loader",
    "UmeAiRT_WirelessImageLoader": "Wireless Image Loader",
    "UmeAiRT_SourceImage_Output": "Wireless Source Image",
    "UmeAiRT_WirelessInpaintComposite": "Wireless Inpaint Composite",
    "UmeAiRT_Label": "Label",
    "UmeAiRT_Signature": "UmeAiRT Signature",
    "UmeAiRT_WirelessImageProcess": "Wireless Image Process",
    "UmeAiRT_GenerationSettings": "Generation Settings (Block)",
    "UmeAiRT_FilesSettings_Checkpoint": "Model Loader (Block)",
    "UmeAiRT_FilesSettings_Checkpoint_Advanced": "Model Loader - Advanced (Block)",
    "UmeAiRT_FilesSettings_FLUX": "Model Loader - FLUX (Block)",
    "UmeAiRT_FilesSettings_Fragmented": "Model Loader (Fragmented)",
    "UmeAiRT_FilesSettings_ZIMG": "Model Loader (Z-IMG)",
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
    "UmeAiRT_Detailer_Daemon_Simple": "Detailer Daemon (Simple)",
    "UmeAiRT_Detailer_Daemon_Advanced": "Detailer Daemon (Advanced)",
    "UmeAiRT_Unpack_ImageBundle": "Unpack Image Bundle",
    "UmeAiRT_Unpack_FilesBundle": "Unpack Models Bundle",
    "UmeAiRT_Unpack_SettingsBundle": "Unpack Settings Bundle",
    "UmeAiRT_Unpack_PromptsBundle": "Unpack Prompts Bundle",
    "UmeAiRT_ControlNetImageApply_Simple": "ControlNet Apply (Simple)",
    "UmeAiRT_ControlNetImageApply_Advanced": "ControlNet Apply (Advanced)",
    "UmeAiRT_ControlNetImageProcess": "ControlNet Process (Unified)",
    "UmeAiRT_Faces_Unpack_Node": "Unpack Faces",
    "UmeAiRT_Tags_Unpack_Node": "Unpack Tags",
    "UmeAiRT_Pipe_Unpack_Node": "Unpack Pipe",
    
    # Tools
    "UmeAiRT_Bundle_Downloader": "💾 Bundle Model Downloader",
    "UmeAiRT_BundleLoader": "📦 Bundle Auto-Loader",
    "UmeAiRT_Log_Viewer": "📜 UmeAiRT Log Viewer",
    "UmeAiRT_HealthCheck": "🩺 UmeAiRT Health Check",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Startup Logging & Optimization Check
from .modules.common import log_node
from .modules.optimization_utils import check_optimizations
import colorama
from colorama import Fore, Style
import server
from aiohttp import web

# Initialize Colorama
colorama.init(convert=True, autoreset=True)

# Define API Route for Signature Image
@server.PromptServer.instance.routes.get("/umeairt/signature")
async def get_signature(request):
    signature_path = os.path.join(os.path.dirname(__file__), "assets", "signature.png")
    if os.path.exists(signature_path):
        return web.FileResponse(signature_path)
    return web.Response(status=404, text="Signature not found")

# 1. Print Node List

n_nodes = len(NODE_CLASS_MAPPINGS)
log_node(f"🧩 Loaded {n_nodes} nodes.", color="RESET")

# 2. Run Optimization Checks
try:
    check_optimizations()
except Exception as e:
    log_node(f"Optimization check failed: {e}", color="RED")

# 3. Final Summary
log_node(f"✅ Initialization Complete.", color="GREEN")
