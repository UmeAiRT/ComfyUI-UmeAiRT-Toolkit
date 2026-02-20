import comfy.samplers
import comfy.utils
from .common import (
    UME_SHARED_STATE, KEY_IMAGESIZE, KEY_FPS, KEY_STEPS, KEY_DENOISE,
    KEY_SEED, KEY_SCHEDULER, KEY_SAMPLER, KEY_CFG, KEY_POSITIVE, KEY_NEGATIVE
)

# --- GUIDANCE NODES ---

class UmeAiRT_Guidance_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "guidance": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1, "display": "slider",
                    "tooltip": "CFG Scale (Guidance Scale). Higher values follow the prompt more strictly, lower values allow more creativity."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_value"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_value(self, guidance):
        UME_SHARED_STATE[KEY_CFG] = guidance
        return ()

class UmeAiRT_Guidance_Output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {} 
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cfg",)
    FUNCTION = "get_value"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_value(self):
        val = UME_SHARED_STATE.get(KEY_CFG, 8.0)
        return (float(val),)


# --- IMAGE SIZE NODES ---

class UmeAiRT_ImageSize_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider", "tooltip": "Target width of the generated image."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider", "tooltip": "Target height of the generated image."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_size"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_size(self, width, height):
        UME_SHARED_STATE[KEY_IMAGESIZE] = {"width": width, "height": height}
        return ()

class UmeAiRT_ImageSize_Output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_size(self):
        val = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
        return (int(val.get("width", 1024)), int(val.get("height", 1024)))


# --- FPS NODES ---

class UmeAiRT_FPS_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 120, "step": 1, "display": "slider",
                    "tooltip": "Frames Per Second for video generation workflows."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, fps):
        UME_SHARED_STATE[KEY_FPS] = fps
        return ()

class UmeAiRT_FPS_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("fps",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_FPS, 24)
        return (int(val),)


# --- STEPS NODES ---

class UmeAiRT_Steps_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 200, "step": 1, "display": "slider",
                    "tooltip": "Number of sampling steps. More steps usually mean higher quality but take longer."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, steps):
        UME_SHARED_STATE[KEY_STEPS] = steps
        return ()

class UmeAiRT_Steps_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_STEPS, 20)
        return (int(val),)


# --- DENOISE NODES ---

class UmeAiRT_Denoise_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "Denoising strength. 1.0 = full new generation (txt2img), <1.0 = modify existing image (img2img)."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, denoise):
        UME_SHARED_STATE[KEY_DENOISE] = denoise
        return ()

class UmeAiRT_Denoise_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("denoise",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_DENOISE, 1.0)
        return (float(val),)


# --- SEED NODES ---

class UmeAiRT_Seed_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for generation. Same seed + same settings = same image."
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, seed):
        UME_SHARED_STATE[KEY_SEED] = seed
        return ()

class UmeAiRT_Seed_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_SEED, 0)
        return (int(val),)


# --- SCHEDULER NODES ---

class UmeAiRT_Scheduler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler algorithm (e.g. normal, karras, exponential)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, scheduler):
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()

class UmeAiRT_Scheduler_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        # Default to first scheduler if missing
        default = comfy.samplers.KSampler.SCHEDULERS[0] if comfy.samplers.KSampler.SCHEDULERS else "normal"
        val = UME_SHARED_STATE.get(KEY_SCHEDULER, default)
        return (val,)


# --- SAMPLER NODES ---

class UmeAiRT_Sampler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm (e.g. euler, dpmpp_2m)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, sampler_name):
        UME_SHARED_STATE[KEY_SAMPLER] = sampler_name
        return ()

class UmeAiRT_Sampler_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        # Default to first sampler if missing
        default = comfy.samplers.KSampler.SAMPLERS[0] if comfy.samplers.KSampler.SAMPLERS else "euler"
        val = UME_SHARED_STATE.get(KEY_SAMPLER, default)
        return (val,)


class UmeAiRT_SamplerScheduler_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler algorithm."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, sampler, scheduler):
        UME_SHARED_STATE[KEY_SAMPLER] = sampler
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()


# --- POSITIVE PROMPT NODES ---

class UmeAiRT_Positive_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The positive prompt describing what you WANT to see in the image."}),
            }
        }

    RETURN_TYPES = ("POSITIVE",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, positive):
        UME_SHARED_STATE[KEY_POSITIVE] = positive
        return (positive,)

class UmeAiRT_Positive_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("POSITIVE",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_POSITIVE, "")
        return (str(val),)


# --- NEGATIVE PROMPT NODES ---

class UmeAiRT_Negative_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative": ("STRING", {"default": "text, watermark", "multiline": True, "dynamicPrompts": True, "tooltip": "The negative prompt describing what you DO NOT WANT to see (e.g. bad quality, watermark)."}),
            }
        }

    RETURN_TYPES = ("NEGATIVE",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, negative):
        UME_SHARED_STATE[KEY_NEGATIVE] = negative
        return (negative,)

class UmeAiRT_Negative_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("NEGATIVE",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_NEGATIVE, "")
        return (str(val),)

# Aliases for backward compatibility and init mapping
UmeAiRT_GlobalSeed = UmeAiRT_Seed_Input
UmeAiRT_Resolution = UmeAiRT_ImageSize_Input
UmeAiRT_Prompt = UmeAiRT_Positive_Input
UmeAiRT_SpeedMode = UmeAiRT_FPS_Input

