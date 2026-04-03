import torch
import numpy as np
import cv2

def apply_dwpose(image_tensor: torch.Tensor, det_model_path: str, pose_model_path: str) -> torch.Tensor:
    import comfy.model_management as mm
    
    from .networks.dwpose.wholebody import Wholebody
    from .networks.dwpose.__init__ import DwposeDetector
    
    device = mm.get_torch_device()
    
    t = Wholebody(det_model_path, pose_model_path, torchscript_device=device)
    detector = DwposeDetector(t)

    out_batch = []
    
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        if img_np.shape[-1] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        with torch.no_grad():
            res_img = detector(img_np, 
                              detect_resolution=512, 
                              include_body=True, 
                              include_hand=True, 
                              include_face=True, 
                              output_type="np", 
                              upscale_method="INTER_CUBIC",
                              xinsr_stick_scaling=True)

        out_tensor = torch.from_numpy(res_img).float() / 255.0
        out_batch.append(out_tensor)

    del detector
    mm.soft_empty_cache()

    return torch.stack(out_batch, dim=0)
