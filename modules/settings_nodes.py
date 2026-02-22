import comfy.samplers
import comfy.utils
from .common import (
    UME_SHARED_STATE, KEY_IMAGESIZE, KEY_FPS, KEY_STEPS, KEY_DENOISE,
    KEY_SEED, KEY_SCHEDULER, KEY_SAMPLER, KEY_CFG, KEY_POSITIVE, KEY_NEGATIVE
)

# --- GENERIC BASE CLASSES & FACTORIES ---

class GenericInputNode:
    """Base class for setting values in the Wireless State."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {cls.INPUT_KEY: cls.INPUT_WIDGET}}

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    OUTPUT_NODE = True

    def set_val(self, **kwargs):
        val = kwargs.get(self.INPUT_KEY)
        UME_SHARED_STATE[self.STATE_KEY] = val
        return ()

class GenericOutputNode:
    """Base class for retrieving values from the Wireless State."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    FUNCTION = "get_val"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(self.STATE_KEY, self.DEFAULT_VAL)
        return (self.RETURN_CAST(val) if self.RETURN_CAST else val,)

def create_input_node(name, state_key, input_key, widget, tooltip, category="UmeAiRT/Wireless/Variables"):
    """Dynamically creates an Input Node class."""
    return type(name, (GenericInputNode,), {
        "__doc__": f"Node to set the global {tooltip.split('.')[0]} in the Wireless State.",
        "INPUT_KEY": input_key,
        "STATE_KEY": state_key,
        "INPUT_WIDGET": widget,
        "CATEGORY": category,
    })

def create_output_node(name, state_key, return_type, return_name, default_val, cast_func=None, category="UmeAiRT/Wireless/Variables"):
    """Dynamically creates an Output Node class."""
    return type(name, (GenericOutputNode,), {
        "__doc__": f"Node to retrieve the global {return_name} from the Wireless State.",
        "STATE_KEY": state_key,
        "RETURN_TYPES": (return_type,),
        "RETURN_NAMES": (return_name,),
        "DEFAULT_VAL": default_val,
        "RETURN_CAST": staticmethod(cast_func) if cast_func else None,
        "CATEGORY": category,
    })

# --- DYNAMICALLY GENERATED NODES ---

UmeAiRT_Guidance_Input = create_input_node("UmeAiRT_Guidance_Input", KEY_CFG, "guidance", 
    ("FLOAT", {"default": 8.0, "min": 0.0, "max": 30.0, "step": 0.1, "display": "slider", "tooltip": "CFG Scale (Guidance Scale)."}), "CFG Scale")
UmeAiRT_Guidance_Output = create_output_node("UmeAiRT_Guidance_Output", KEY_CFG, "FLOAT", "cfg", 8.0, float)

UmeAiRT_FPS_Input = create_input_node("UmeAiRT_FPS_Input", KEY_FPS, "fps", 
    ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "display": "slider", "tooltip": "Frames Per Second."}), "FPS")
UmeAiRT_FPS_Output = create_output_node("UmeAiRT_FPS_Output", KEY_FPS, "INT", "fps", 24, int)

UmeAiRT_Steps_Input = create_input_node("UmeAiRT_Steps_Input", KEY_STEPS, "steps", 
    ("INT", {"default": 20, "min": 1, "max": 200, "step": 1, "display": "slider", "tooltip": "Number of sampling steps."}), "Steps")
UmeAiRT_Steps_Output = create_output_node("UmeAiRT_Steps_Output", KEY_STEPS, "INT", "steps", 20, int)

UmeAiRT_Denoise_Input = create_input_node("UmeAiRT_Denoise_Input", KEY_DENOISE, "denoise", 
    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "tooltip": "Denoising strength."}), "Denoise")
UmeAiRT_Denoise_Output = create_output_node("UmeAiRT_Denoise_Output", KEY_DENOISE, "FLOAT", "denoise", 1.0, float)

UmeAiRT_Seed_Input = create_input_node("UmeAiRT_Seed_Input", KEY_SEED, "seed", 
    ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed."}), "Seed")
UmeAiRT_Seed_Output = create_output_node("UmeAiRT_Seed_Output", KEY_SEED, "INT", "seed", 0, int)

UmeAiRT_Scheduler_Input = create_input_node("UmeAiRT_Scheduler_Input", KEY_SCHEDULER, "scheduler", 
    (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler algorithm."}), "Scheduler")
UmeAiRT_Scheduler_Output = create_output_node("UmeAiRT_Scheduler_Output", KEY_SCHEDULER, "*", "scheduler", 
    comfy.samplers.KSampler.SCHEDULERS[0] if comfy.samplers.KSampler.SCHEDULERS else "normal")

UmeAiRT_Sampler_Input = create_input_node("UmeAiRT_Sampler_Input", KEY_SAMPLER, "sampler_name", 
    (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}), "Sampler")
UmeAiRT_Sampler_Output = create_output_node("UmeAiRT_Sampler_Output", KEY_SAMPLER, "*", "sampler_name", 
    comfy.samplers.KSampler.SAMPLERS[0] if comfy.samplers.KSampler.SAMPLERS else "euler")

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
        UME_SHARED_STATE[KEY_SAMPLER] = sampler
        UME_SHARED_STATE[KEY_SCHEDULER] = scheduler
        return ()

# Pass-through input nodes (return what they set)
class GenericPassThroughNode(GenericInputNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {cls.INPUT_KEY: cls.INPUT_WIDGET}}
    
    def set_val(self, **kwargs):
        val = kwargs.get(self.INPUT_KEY)
        UME_SHARED_STATE[self.STATE_KEY] = val
        return (val,)

def create_passthrough_node(name, state_key, input_key, widget, tooltip, return_type, return_name, category="UmeAiRT/Wireless/Variables"):
    return type(name, (GenericPassThroughNode,), {
        "__doc__": f"Node to set and pass through the global {tooltip.split('.')[0]} in the Wireless State.",
        "INPUT_KEY": input_key,
        "STATE_KEY": state_key,
        "INPUT_WIDGET": widget,
        "RETURN_TYPES": (return_type,),
        "RETURN_NAMES": (return_name,),
        "CATEGORY": category,
    })

UmeAiRT_Positive_Input = create_passthrough_node("UmeAiRT_Positive_Input", KEY_POSITIVE, "positive", 
    ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Positive prompt."}), "Positive prompt", "POSITIVE", "positive")
UmeAiRT_Positive_Output = create_output_node("UmeAiRT_Positive_Output", KEY_POSITIVE, "POSITIVE", "positive", "", str)

UmeAiRT_Negative_Input = create_passthrough_node("UmeAiRT_Negative_Input", KEY_NEGATIVE, "negative", 
    ("STRING", {"default": "text, watermark", "multiline": True, "dynamicPrompts": True, "tooltip": "Negative prompt."}), "Negative prompt", "NEGATIVE", "negative")
UmeAiRT_Negative_Output = create_output_node("UmeAiRT_Negative_Output", KEY_NEGATIVE, "NEGATIVE", "negative", "", str)

# Image size nodes (multiple inputs/outputs)
class UmeAiRT_ImageSize_Input:
    """Node to set the global Image Size (Width and Height) in the Wireless State."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider", "tooltip": "Target width."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8, "display": "slider", "tooltip": "Target height."}),
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
    """Node to retrieve the global Image Size (Width and Height) from the Wireless State."""
    @classmethod
    def INPUT_TYPES(s): return {"required": {}}
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs): return float("nan")

    def get_size(self):
        val = UME_SHARED_STATE.get(KEY_IMAGESIZE, {"width": 1024, "height": 1024})
        return (int(val.get("width", 1024)), int(val.get("height", 1024)))

# Aliases for backward compatibility and init mapping
UmeAiRT_GlobalSeed = UmeAiRT_Seed_Input
UmeAiRT_Resolution = UmeAiRT_ImageSize_Input
UmeAiRT_Prompt = UmeAiRT_Positive_Input
UmeAiRT_SpeedMode = UmeAiRT_FPS_Input

