import folder_paths
import nodes as comfy_nodes
from .logger import log_node


class UmeAiRT_MultiLoraLoader:
    """Wired Multi-LoRA loader. Applies up to 3 LoRAs to a Model+CLIP pair."""
    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The base AI model to apply LoRAs to."}),
                "clip": ("CLIP", {"tooltip": "The text encoder model (handles your prompts)."}),
                
                # SLOT 1
                "lora_1": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA slot 1."}),
                "lora_1_name": (lora_list, {"tooltip": "Select LoRA model."}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": "LoRA strength (1.0 = standard)."}),
                
                # SLOT 2
                "lora_2": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA slot 2."}),
                "lora_2_name": (lora_list, {"tooltip": "Select LoRA model."}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": "LoRA strength."}),
                
                # SLOT 3
                "lora_3": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF", "tooltip": "Enable LoRA slot 3."}),
                "lora_3_name": (lora_list, {"tooltip": "Select LoRA model."}),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "tooltip": "LoRA strength."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "UmeAiRT/Loaders"

    def load_loras(self, model, clip, 
                   lora_1, lora_1_name, lora_1_strength,
                   lora_2, lora_2_name, lora_2_strength,
                   lora_3, lora_3_name, lora_3_strength):

        def apply_lora(curr_model, curr_clip, is_on, name, strength):
            if is_on and name != "None":
                return self.lora_loader.load_lora(curr_model, curr_clip, name, strength, strength)
            return curr_model, curr_clip
        
        m, c = apply_lora(model, clip, lora_1, lora_1_name, lora_1_strength)
        m, c = apply_lora(m, c, lora_2, lora_2_name, lora_2_strength)
        m, c = apply_lora(m, c, lora_3, lora_3_name, lora_3_strength)

        return (m, c)
