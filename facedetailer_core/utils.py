import torch
import torchvision
import cv2
import numpy as np
import folder_paths
import nodes
from PIL import Image
import comfy.model_management
import torch.nn.functional as F
import math

class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)


def tensor_convert_rgba(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")


def resize_with_padding(image, target_w: int, target_h: int):
    _tensor_check_image(image)
    b, h, w, c = image.shape
    image = image.permute(0, 3, 1, 2)  # B, C, H, W

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    image = image.permute(0, 2, 3, 1)  # B, H, W, C
    return image, (pad_top, pad_bottom, pad_left, pad_right)


def remove_padding(image, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    return image[:, pad_top:image.shape[1] - pad_bottom, pad_left:image.shape[2] - pad_right, :]


def adjust_bbox_after_resize(bbox, original_size, target_size, padding):
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    pad_top, pad_bottom, pad_left, pad_right = padding

    scale = min(target_w / orig_w, target_h / orig_h)

    # Apply scale
    x1 = int(bbox[0] * scale + pad_left)
    y1 = int(bbox[1] * scale + pad_top)
    x2 = int(bbox[2] * scale + pad_left)
    y2 = int(bbox[3] * scale + pad_top)

    return x1, y1, x2, y2


def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)


def tensor_get_size(image):
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def numpy2pil(image):
    return Image.fromarray(np.clip(255. * image.squeeze(0), 0, 255).astype(np.uint8))

def to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        return tensor2pil(image)
    if isinstance(image, np.ndarray):
        return numpy2pil(image)
    raise ValueError(f"Cannot convert {type(image)} to PIL.Image")

def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")

def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    if isinstance(image, torch.Tensor):
        return image.numpy()
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Cannot convert {type(image)} to numpy.ndarray")

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]


def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return


def tensor_crop(image, crop_region):
    _tensor_check_image(image)
    return crop_ndarray4(image, crop_region)


def tensor2numpy(image):
    _tensor_check_image(image)
    return image.numpy()


def tensor_paste(image1, image2, left_top, mask):
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)

    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)

    x, y = left_top
    _, h1, w1, c1 = image1.shape
    _, h2, w2, c2 = image2.shape

    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]

    region1 = image1[:, y:y+h, x:x+w, :]
    region2 = image2[:, :h, :w, :]

    if c1 == 3 and c2 == 3:
        image1[:, y:y+h, x:x+w, :] = (1 - mask) * region1 + mask * region2

    elif c1 == 4 and c2 == 4:
        image1[:, y:y+h, x:x+w, :3] = (
            (1 - mask) * region1[:, :, :, :3] +
            mask * region2[:, :, :, :3]
        )
        a1 = region1[:, :, :, 3:4]
        a2 = region2[:, :, :, 3:4] * mask
        new_alpha = a1 + a2 * (1 - a1)
        image1[:, y:y+h, x:x+w, 3:4] = new_alpha

    elif c1 == 4 and c2 == 3:
        image1[:, y:y+h, x:x+w, :3] = (
            (1 - mask) * region1[:, :, :, :3] +
            mask * region2
        )
        image1[:, y:y+h, x:x+w, 3:4] = region1[:, :, :, 3:4] * (1 - mask) + mask

    elif c1 == 3 and c2 == 4:
        effective_mask = mask * region2[:, :, :, 3:4]
        image1[:, y:y+h, x:x+w, :] = (
            (1 - effective_mask) * region1 +
            effective_mask * region2[:, :, :, :3]
        )
    return



def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return make_2d_mask(mask)

    mask = make_2d_mask(mask)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if dilation_factor > 0:
        result = cv2.dilate(mask_np, kernel, iterations=iter)
    else:
        result = cv2.erode(mask_np, kernel, iterations=iter)

    return torch.from_numpy(result)

def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)
    return mask

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask

def make_4d_mask(mask):
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    return mask

def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized_mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    return resized_mask.squeeze(0)

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor
    
    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)
    
    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2
    
    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)
    
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)
    
    return [new_x1, new_y1, new_x2, new_y2]

def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]
    return npimg[:, y1:y2, x1:x2, :]

def crop_ndarray3(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]
    return npimg[y1:y2, x1:x2, :]

def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]
    return npimg[y1:y2, x1:x2]

def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = int(kernel_size)
    kernel_size = kernel_size*2+1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest/2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask 

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask

def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]

    # We use direct VAE Encode from nodes.py or comfy.sd if available, 
    # but here we can just use the VAE object passed in.
    # VAE.encode returns a latent representation. 
    # ComfyUI nodes.VAEEncode wraps it.
    
    # Simple direct encoding
    t = vae.encode(pixels[:,:,:,:3])
    return {"samples": t}
