import torch
import numpy as np
from PIL import Image

def apply_zoedepth(image_tensor: torch.Tensor, model_path: str) -> torch.Tensor:
    """
    Applies ZoeDepth using HuggingFace transformers locally.
    Expected input: [B, H, W, C] normalized to [0, 1].
    Output: [B, H, W, C] normalized to [0, 1] containing the depth map.
    """
    import comfy.model_management as mm
    from transformers import pipeline, AutoImageProcessor, ZoeDepthForDepthEstimation
    
    device = mm.get_torch_device()
    
    # Load model and processor pointing to local UmeAiRT asset path
    image_processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = ZoeDepthForDepthEstimation.from_pretrained(model_path, local_files_only=True)
    
    # Put model on GPU
    model.to(device)
    model.eval()
    
    pipe = pipeline(task="depth-estimation", model=model, image_processor=image_processor, device=device)
    
    out_batch = []
    
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        with torch.no_grad():
            result = pipe(pil_img)
            depth_img = result["depth"]
            
        depth_array = np.array(depth_img, dtype=np.float32)
        
        # Normalize depth map
        vmin = np.percentile(depth_array, 2)
        vmax = np.percentile(depth_array, 85)
        depth_array = depth_array - vmin
        
        # Prevent division by zero
        if vmax - vmin > 0:
            depth_array = depth_array / (vmax - vmin)
            
        depth_array = 1.0 - depth_array
        depth_array = (depth_array * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert to 3 channels [H, W, 3]
        if depth_array.ndim == 2:
            import cv2
            depth_3c = cv2.cvtColor(depth_array, cv2.COLOR_GRAY2RGB)
        else:
            depth_3c = depth_array
            
        out_tensor = torch.from_numpy(depth_3c).float() / 255.0
        out_batch.append(out_tensor)
        
    # Free VRAM explicitly
    del pipe
    del model
    mm.soft_empty_cache()
        
    return torch.stack(out_batch, dim=0)
