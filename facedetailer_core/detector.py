import os
import torch
import numpy as np
import folder_paths
from . import utils
from .logic import SEG
import sys

# Try to import ultralytics
try:
    from ultralytics import YOLO
    has_ultralytics = True
except ImportError:
    has_ultralytics = False

class UmeAiRT_BboxDetector:
    def __init__(self, model_path):
        if not has_ultralytics:
            raise ImportError("UmeAiRT: 'ultralytics' library is required for BBOX detection. Please install it via 'pip install ultralytics'.")
        
        self.model_path = model_path
        # Load model on CPU initially, move to GPU during inference if needed/possible
        self.model = YOLO(model_path) 

    def detect(self, image, threshold, dilation, crop_factor, drop_size):
        # image is [B, H, W, C] tensor
        
        # Prepare results container
        all_segs = []
        
        for i in range(len(image)):
            img_tensor = image[i] # [H, W, C]
            
            # Convert to numpy uint8 for YOLO
            img_np = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            
            # Inference
            # YOLO returns a list of Results objects
            results = self.model.predict(img_np, conf=threshold, verbose=False)
            
            h, w = img_np.shape[:2]
            
            img_segs = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # bounding box data
                    # box.xyxy is [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    label_idx = int(box.cls[0].item())
                    label = self.model.names[label_idx]
                    
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Apply crop factor logic (from impact core logic)
                    crop_region = utils.make_crop_region(w, h, bbox, crop_factor)
                    
                    # Create mask for this bbox (simple rectangle)
                    # Implementation detail: Impact Pack BBOX detector creates a mask for the bbox area
                    mask = np.zeros((h, w), dtype=np.float32)
                    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
                    
                    # Apply dilation to the mask if needed (though crop_region is usually what matters for detailing)
                    if dilation != 0:
                        # Logic to dilate the rectangular mask
                        kernel = np.ones((abs(dilation), abs(dilation)), np.uint8)
                        if dilation > 0:
                            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                            mask = utils.dilate_mask(mask, dilation).squeeze().numpy()
                        else:
                             # Erosion
                            pass 

                    # Crop Image & Mask
                    cropped_img = utils.crop_ndarray3(img_np, crop_region) # Returns numpy
                    cropped_mask = utils.crop_ndarray2(mask, crop_region)
                    
                    # Check drop size
                    if (crop_region[2] - crop_region[0]) < drop_size or (crop_region[3] - crop_region[1]) < drop_size:
                        continue
                        
                    # Create SEG
                    seg = SEG(
                        cropped_image=cropped_img, # numpy
                        cropped_mask=cropped_mask, # numpy
                        confidence=conf,
                        crop_region=crop_region,
                        bbox=bbox,
                        label=label,
                        control_net_wrapper=None
                    )
                    img_segs.append(seg)
            
            all_segs.extend(img_segs)

        # structure: (image_shape, list_of_segs)
        return (image.shape, all_segs)

def load_bbox_model(model_name):
    path = folder_paths.get_full_path("bbox", model_name)
    if not path:
        # Fallback check standard locations if 'bbox' is not registered or found
        path = folder_paths.get_full_path("ultralytics", model_name)
    
    if not path:
         raise FileNotFoundError(f"Model {model_name} not found in 'bbox' or 'ultralytics' folders.")
         
    return UmeAiRT_BboxDetector(path)
