from PIL import Image
from usdu_utils import pil_to_tensor, tensor_to_pil
from . import shared
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def upscale(self, img: Image, scale, selected_model: str = None):
        if scale == 1.0:
            return img
        if (shared.actual_upscaler is None):
            return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
        if "execute" in dir(ImageUpscaleWithModel):  
            # V3 schema: https://github.com/comfyanonymous/ComfyUI/pull/10149
            (upscaled,) = ImageUpscaleWithModel.execute(shared.actual_upscaler, shared.batch_as_tensor)
        else:
            (upscaled,) = ImageUpscaleWithModel().upscale(shared.actual_upscaler, shared.batch_as_tensor)
        shared.batch = [tensor_to_pil(upscaled, i) for i in range(len(upscaled))]
        return shared.batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
