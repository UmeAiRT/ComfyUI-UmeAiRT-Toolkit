# --- Dynamic Sampler Registration ---
try:
    from .modules.extra_samplers import register_extra_samplers
    register_extra_samplers()
except Exception as e:
    print(f"[UmeAiRT-Toolkit] Failed to register extra samplers: {e}")
# ------------------------------------

from .modules.model_nodes import (
    UmeAiRT_MultiLoraLoader
)
from .modules.image_nodes import (
    UmeAiRT_SourceImage_Output,
    UmeAiRT_PipelineInpaintComposite,
    UmeAiRT_PipelineImageSaver
)

# Register 'bbox' folder for FaceDetailer
import folder_paths
import os
try:
    folder_paths.add_model_folder_path("bbox", os.path.join(folder_paths.models_dir, "bbox"))
except Exception as e:
    print(f"[UmeAiRT-Toolkit] bbox folder registration note: {e}")
if "bbox" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["bbox"] = ([os.path.join(folder_paths.models_dir, "bbox")], folder_paths.supported_pt_extensions)

from .modules.logic_nodes import (
    UmeAiRT_PipelineUltimateUpscale,
    UmeAiRT_PipelineSeedVR2Upscale,
    UmeAiRT_PipelineSubjectDetailer,
    UmeAiRT_Detailer_Daemon
)
from .modules.block_nodes import (
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3, UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    UmeAiRT_ControlNetImageApply, UmeAiRT_ControlNetImageProcess,
    UmeAiRT_GenerationSettings,
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_FilesSettings_Fragmented,
    UmeAiRT_FilesSettings_ZIMG,
    UmeAiRT_BlockImageLoader, UmeAiRT_BlockImageProcess,
    UmeAiRT_ImageProcess_Img2Img, UmeAiRT_ImageProcess_Inpaint, UmeAiRT_ImageProcess_Outpaint,
    UmeAiRT_BlockSampler,
    UmeAiRT_BundleLoader,
    UmeAiRT_Positive_Input, UmeAiRT_Negative_Input
)
from .modules.utils_nodes import (
    UmeAiRT_Label,
    UmeAiRT_Bundle_Downloader,
    UmeAiRT_Log_Viewer,
    UmeAiRT_Unpack_Settings,
    UmeAiRT_Unpack_Prompt,
    UmeAiRT_Faces_Unpack_Node,
    UmeAiRT_Tags_Unpack_Node,
    UmeAiRT_Pipe_Unpack_Node,
    UmeAiRT_Unpack_SettingsBundle,
    UmeAiRT_Unpack_PromptsBundle,
    UmeAiRT_Unpack_ImageBundle,
    UmeAiRT_Unpack_FilesBundle,
    UmeAiRT_Unpack_Pipeline,
    UmeAiRT_Pack_Bundle,
    UmeAiRT_Signature,
    UmeAiRT_HealthCheck
)

NODE_CLASS_MAPPINGS = {
    # Block Loaders
    "UmeAiRT_FilesSettings_Checkpoint": UmeAiRT_FilesSettings_Checkpoint,
    "UmeAiRT_FilesSettings_FLUX": UmeAiRT_FilesSettings_FLUX,
    "UmeAiRT_FilesSettings_Fragmented": UmeAiRT_FilesSettings_Fragmented,
    "UmeAiRT_FilesSettings_ZIMG": UmeAiRT_FilesSettings_ZIMG,
    "UmeAiRT_BundleLoader": UmeAiRT_BundleLoader,
    "UmeAiRT_MultiLoraLoader": UmeAiRT_MultiLoraLoader,

    # Block Settings & Image
    "UmeAiRT_GenerationSettings": UmeAiRT_GenerationSettings,
    "UmeAiRT_BlockImageLoader": UmeAiRT_BlockImageLoader,
    "UmeAiRT_BlockImageProcess": UmeAiRT_BlockImageProcess,
    "UmeAiRT_ImageProcess_Img2Img": UmeAiRT_ImageProcess_Img2Img,
    "UmeAiRT_ImageProcess_Inpaint": UmeAiRT_ImageProcess_Inpaint,
    "UmeAiRT_ImageProcess_Outpaint": UmeAiRT_ImageProcess_Outpaint,
    "UmeAiRT_LoraBlock_1": UmeAiRT_LoraBlock_1,
    "UmeAiRT_LoraBlock_3": UmeAiRT_LoraBlock_3,
    "UmeAiRT_LoraBlock_5": UmeAiRT_LoraBlock_5,
    "UmeAiRT_LoraBlock_10": UmeAiRT_LoraBlock_10,
    "UmeAiRT_ControlNetImageApply": UmeAiRT_ControlNetImageApply,
    "UmeAiRT_ControlNetImageProcess": UmeAiRT_ControlNetImageProcess,

    # Prompt Editors
    "UmeAiRT_Positive_Input": UmeAiRT_Positive_Input,
    "UmeAiRT_Negative_Input": UmeAiRT_Negative_Input,

    # Sampler & Post-Process (Pipeline)
    "UmeAiRT_BlockSampler": UmeAiRT_BlockSampler,
    "UmeAiRT_PipelineUltimateUpscale": UmeAiRT_PipelineUltimateUpscale,
    "UmeAiRT_PipelineSeedVR2Upscale": UmeAiRT_PipelineSeedVR2Upscale,
    "UmeAiRT_PipelineSubjectDetailer": UmeAiRT_PipelineSubjectDetailer,
    "UmeAiRT_Detailer_Daemon": UmeAiRT_Detailer_Daemon,

    # Image
    "UmeAiRT_SourceImage_Output": UmeAiRT_SourceImage_Output,
    "UmeAiRT_PipelineInpaintComposite": UmeAiRT_PipelineInpaintComposite,
    "UmeAiRT_PipelineImageSaver": UmeAiRT_PipelineImageSaver,

    # Pack/Unpack (Interoperability)
    "UmeAiRT_Pack_Bundle": UmeAiRT_Pack_Bundle,
    "UmeAiRT_Unpack_Pipeline": UmeAiRT_Unpack_Pipeline,
    "UmeAiRT_Unpack_FilesBundle": UmeAiRT_Unpack_FilesBundle,
    "UmeAiRT_Unpack_ImageBundle": UmeAiRT_Unpack_ImageBundle,
    "UmeAiRT_Unpack_SettingsBundle": UmeAiRT_Unpack_SettingsBundle,
    "UmeAiRT_Unpack_PromptsBundle": UmeAiRT_Unpack_PromptsBundle,
    "UmeAiRT_Faces_Unpack_Node": UmeAiRT_Faces_Unpack_Node,
    "UmeAiRT_Tags_Unpack_Node": UmeAiRT_Tags_Unpack_Node,
    "UmeAiRT_Pipe_Unpack_Node": UmeAiRT_Pipe_Unpack_Node,

    # Utils
    "UmeAiRT_Label": UmeAiRT_Label,
    "UmeAiRT_Signature": UmeAiRT_Signature,
    "UmeAiRT_Bundle_Downloader": UmeAiRT_Bundle_Downloader,
    "UmeAiRT_Log_Viewer": UmeAiRT_Log_Viewer,
    "UmeAiRT_HealthCheck": UmeAiRT_HealthCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "UmeAiRT_FilesSettings_Checkpoint": "⬡ Checkpoint Loader",
    "UmeAiRT_FilesSettings_FLUX": "⬡ FLUX Loader",
    "UmeAiRT_FilesSettings_Fragmented": "⬡ Fragmented Loader",
    "UmeAiRT_FilesSettings_ZIMG": "⬡ Z-IMG Loader",
    "UmeAiRT_BundleLoader": "⬡ 📦 Bundle Auto-Loader",
    "UmeAiRT_MultiLoraLoader": "⬡ Multi-LoRA Loader",

    # Settings & Image
    "UmeAiRT_GenerationSettings": "⬡ Generation Settings",
    "UmeAiRT_BlockImageLoader": "⬡ Image Loader",
    "UmeAiRT_BlockImageProcess": "⬡ Image Process",
    "UmeAiRT_ImageProcess_Img2Img": "⬡ Image Process (Img2Img)",
    "UmeAiRT_ImageProcess_Inpaint": "⬡ Image Process (Inpaint)",
    "UmeAiRT_ImageProcess_Outpaint": "⬡ Image Process (Outpaint)",
    "UmeAiRT_LoraBlock_1": "⬡ LoRA 1x",
    "UmeAiRT_LoraBlock_3": "⬡ LoRA 3x",
    "UmeAiRT_LoraBlock_5": "⬡ LoRA 5x",
    "UmeAiRT_LoraBlock_10": "⬡ LoRA 10x",
    "UmeAiRT_ControlNetImageApply": "⬡ ControlNet Apply",
    "UmeAiRT_ControlNetImageProcess": "⬡ ControlNet Process",

    # Prompt Editors
    "UmeAiRT_Positive_Input": "⬡ Positive Prompt Input",
    "UmeAiRT_Negative_Input": "⬡ Negative Prompt Input",

    # Sampler & Post-Process
    "UmeAiRT_BlockSampler": "⬡ KSampler",
    "UmeAiRT_PipelineUltimateUpscale": "⬡ UltimateSD Upscale",
    "UmeAiRT_PipelineSeedVR2Upscale": "⬡ SeedVR2 Upscale",
    "UmeAiRT_PipelineSubjectDetailer": "⬡ Subject Detailer",
    "UmeAiRT_Detailer_Daemon": "⬡ Detailer Daemon",

    # Image
    "UmeAiRT_SourceImage_Output": "⬡ Source Image Output",
    "UmeAiRT_PipelineInpaintComposite": "⬡ Inpaint Composite",
    "UmeAiRT_PipelineImageSaver": "⬡ Image Saver",

    # Pack/Unpack
    "UmeAiRT_Pack_Bundle": "⬡ Pack Models Bundle",
    "UmeAiRT_Unpack_Pipeline": "⬡ Unpack Pipeline",
    "UmeAiRT_Unpack_FilesBundle": "⬡ Unpack Models Bundle",
    "UmeAiRT_Unpack_ImageBundle": "⬡ Unpack Image Bundle",
    "UmeAiRT_Unpack_SettingsBundle": "⬡ Unpack Settings Bundle",
    "UmeAiRT_Unpack_PromptsBundle": "⬡ Unpack Prompts Bundle",
    "UmeAiRT_Faces_Unpack_Node": "⬡ Unpack Faces",
    "UmeAiRT_Tags_Unpack_Node": "⬡ Unpack Tags",
    "UmeAiRT_Pipe_Unpack_Node": "⬡ Unpack Pipe",

    # Utils
    "UmeAiRT_Label": "⬡ Label",
    "UmeAiRT_Signature": "⬡ UmeAiRT Signature",
    "UmeAiRT_Bundle_Downloader": "⬡ 💾 Bundle Model Downloader",
    "UmeAiRT_Log_Viewer": "⬡ 📜 UmeAiRT Log Viewer",
    "UmeAiRT_HealthCheck": "⬡ 🩺 UmeAiRT Health Check",
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
log_node("✅ Initialization Complete.", color="GREEN")
