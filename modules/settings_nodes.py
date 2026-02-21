import comfy.samplers
import comfy.utils
from .common import (
    UME_SHARED_STATE, KEY_IMAGESIZE, KEY_FPS, KEY_STEPS, KEY_DENOISE,
    KEY_SEED, KEY_SCHEDULER, KEY_SAMPLER, KEY_CFG, KEY_POSITIVE, KEY_NEGATIVE
)

# --- GUIDANCE NODES ---

class UmeAiRT_Guidance_Input:
    """Node to set the global CFG Scale (Guidance Scale) value in the Wireless State."""
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
        """Updates the global CFG scale value.

        Args:
            guidance (float): The CFG scale to store.

        Returns:
            tuple: An empty tuple as this is an output node.
        """
        UME_SHARED_STATE[KEY_CFG] = guidance
        return ()

class UmeAiRT_Guidance_Output:
    """Node to retrieve the global CFG Scale (Guidance Scale) value from the Wireless State."""
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
        """Fetches the global CFG scale value.

        Returns:
            tuple: A tuple containing the CFG scale (float). Defaults to 8.0 if not set.
        """
        val = UME_SHARED_STATE.get(KEY_CFG, 8.0)
        return (float(val),)


# --- IMAGE SIZE NODES ---

class UmeAiRT_ImageSize_Input:
    """Node to set the global Image Size (Width and Height) in the Wireless State."""
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
        """Updates the global image dimensions.

        Args:
            width (int): The target width in pixels.
            height (int): The target height in pixels.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_IMAGESIZE] = {"width": width, "height": height}
        return ()

class UmeAiRT_ImageSize_Output:
    """Node to retrieve the global Image Size (Width and Height) from the Wireless State."""
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
        """Fetches the global image dimensions.

        Returns:
            tuple: A tuple containing (width, height) as integers. Defaults to (1024, 1024).
        """
        val = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
        return (int(val.get("width", 1024)), int(val.get("height", 1024)))


# --- FPS NODES ---

class UmeAiRT_FPS_Input:
    """Node to set the global Frames Per Second (FPS) value in the Wireless State."""
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
        """Updates the global FPS value.

        Args:
            fps (int): The target frames per second.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_FPS] = fps
        return ()

class UmeAiRT_FPS_Output:
    """Node to retrieve the global Frames Per Second (FPS) value from the Wireless State."""
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
        """Fetches the global FPS value.

        Returns:
            tuple: A tuple containing the FPS (int). Defaults to 24.
        """
        val = UME_SHARED_STATE.get(KEY_FPS, 24)
        return (int(val),)


# --- STEPS NODES ---

class UmeAiRT_Steps_Input:
    """Node to set the global Sampling Steps value in the Wireless State."""
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
        """Updates the global sampling steps.

        Args:
            steps (int): The number of steps for generation.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_STEPS] = steps
        return ()

class UmeAiRT_Steps_Output:
    """Node to retrieve the global Sampling Steps value from the Wireless State."""
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
        """Fetches the global sampling steps value.

        Returns:
            tuple: A tuple containing the steps (int). Defaults to 20.
        """
        val = UME_SHARED_STATE.get(KEY_STEPS, 20)
        return (int(val),)


# --- DENOISE NODES ---

class UmeAiRT_Denoise_Input:
    """Node to set the global Denoise strength value in the Wireless State."""
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
        """Updates the global denoise strength.

        Args:
            denoise (float): The denoise value (0.0 to 1.0).

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_DENOISE] = denoise
        return ()

class UmeAiRT_Denoise_Output:
    """Node to retrieve the global Denoise strength value from the Wireless State."""
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
        """Fetches the global denoise strength value.

        Returns:
            tuple: A tuple containing the denoise strength (float). Defaults to 1.0.
        """
        val = UME_SHARED_STATE.get(KEY_DENOISE, 1.0)
        return (float(val),)


# --- SEED NODES ---

class UmeAiRT_Seed_Input:
    """Node to set the global Random Seed value in the Wireless State."""
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
        """Updates the global random seed.

        Args:
            seed (int): The generation seed.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_SEED] = seed
        return ()

class UmeAiRT_Seed_Output:
    """Node to retrieve the global Random Seed value from the Wireless State."""
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
        """Fetches the global random seed value.

        Returns:
            tuple: A tuple containing the seed (int). Defaults to 0.
        """
        val = UME_SHARED_STATE.get(KEY_SEED, 0)
        return (int(val),)


# --- SCHEDULER NODES ---

class UmeAiRT_Scheduler_Input:
    """Node to set the global Scheduler algorithm in the Wireless State."""
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
        """Updates the global scheduler algorithm.

        Args:
            scheduler (str): The name of the scheduler.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()

class UmeAiRT_Scheduler_Output:
    """Node to retrieve the global Scheduler algorithm from the Wireless State."""
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
        """Fetches the global scheduler algorithm.

        Returns:
            tuple: A tuple containing the scheduler name (str). Defaults to 'normal'.
        """
        # Default to first scheduler if missing
        default = comfy.samplers.KSampler.SCHEDULERS[0] if comfy.samplers.KSampler.SCHEDULERS else "normal"
        val = UME_SHARED_STATE.get(KEY_SCHEDULER, default)
        return (val,)


# --- SAMPLER NODES ---

class UmeAiRT_Sampler_Input:
    """Node to set the global Sampler algorithm in the Wireless State."""
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
        """Updates the global sampler algorithm.

        Args:
            sampler_name (str): The name of the sampler.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_SAMPLER] = sampler_name
        return ()

class UmeAiRT_Sampler_Output:
    """Node to retrieve the global Sampler algorithm from the Wireless State."""
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
        """Fetches the global sampler algorithm.

        Returns:
            tuple: A tuple containing the sampler name (str). Defaults to 'euler'.
        """
        # Default to first sampler if missing
        default = comfy.samplers.KSampler.SAMPLERS[0] if comfy.samplers.KSampler.SAMPLERS else "euler"
        val = UME_SHARED_STATE.get(KEY_SAMPLER, default)
        return (val,)


class UmeAiRT_SamplerScheduler_Input:
    """Combo Node to set both Sampler and Scheduler algorithms in the Wireless State."""
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
        """Updates both the global sampler and scheduler algorithms simultaneously.

        Args:
            sampler (str): The name of the sampler.
            scheduler (str): The name of the scheduler.

        Returns:
            tuple: An empty tuple.
        """
        UME_SHARED_STATE[KEY_SAMPLER] = sampler
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()


# --- POSITIVE PROMPT NODES ---

class UmeAiRT_Positive_Input:
    """Node to set the global Positive Prompt text in the Wireless State."""
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
        """Updates the global positive prompt and passes it through.

        Args:
            positive (str): The positive prompt text.

        Returns:
            tuple: A tuple containing the positive prompt string.
        """
        UME_SHARED_STATE[KEY_POSITIVE] = positive
        return (positive,)

class UmeAiRT_Positive_Output:
    """Node to retrieve the global Positive Prompt text from the Wireless State."""
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
        """Fetches the global positive prompt.

        Returns:
            tuple: A tuple containing the positive prompt string. Defaults to empty string.
        """
        val = UME_SHARED_STATE.get(KEY_POSITIVE, "")
        return (str(val),)


# --- NEGATIVE PROMPT NODES ---

class UmeAiRT_Negative_Input:
    """Node to set the global Negative Prompt text in the Wireless State."""
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
        """Updates the global negative prompt and passes it through.

        Args:
            negative (str): The negative prompt text.

        Returns:
            tuple: A tuple containing the negative prompt string.
        """
        UME_SHARED_STATE[KEY_NEGATIVE] = negative
        return (negative,)

class UmeAiRT_Negative_Output:
    """Node to retrieve the global Negative Prompt text from the Wireless State."""
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
        """Fetches the global negative prompt.

        Returns:
            tuple: A tuple containing the negative prompt string. Defaults to empty string.
        """
        val = UME_SHARED_STATE.get(KEY_NEGATIVE, "")
        return (str(val),)

# Aliases for backward compatibility and init mapping
UmeAiRT_GlobalSeed = UmeAiRT_Seed_Input
UmeAiRT_Resolution = UmeAiRT_ImageSize_Input
UmeAiRT_Prompt = UmeAiRT_Positive_Input
UmeAiRT_SpeedMode = UmeAiRT_FPS_Input

