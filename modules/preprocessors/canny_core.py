import torch
import numpy as np

def apply_canny(image_tensor: torch.Tensor, low_threshold: int = 100, high_threshold: int = 200) -> torch.Tensor:
    """
    Applies OpenCV Canny edge detection to a ComfyUI image tensor.
    Expected input: [B, H, W, C] normalized to [0, 1].
    Output: [B, H, W, C] normalized to [0, 1] with black background and white edges.
    """
    import cv2
    
    out_batch = []
    
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        if img_np.shape[-1] >= 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.squeeze(-1) if img_np.ndim == 3 else img_np
            
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert edge map back to 3 channels for ControlNet input
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        out_tensor = torch.from_numpy(edges_3c).float() / 255.0
        out_batch.append(out_tensor)
        
    return torch.stack(out_batch, dim=0)
