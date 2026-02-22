import folder_paths
import nodes as comfy_nodes
from .common import (
    UME_SHARED_STATE, KEY_MODEL, KEY_VAE, KEY_CLIP, KEY_LATENT, 
    KEY_MODEL_NAME, KEY_LORAS
)
from .logger import log_node

# --- MODEL/VAE/CLIP NODES ---

class UmeAiRT_Model_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Model (UNET)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, model):
        UME_SHARED_STATE[KEY_MODEL] = model
        return ()

class UmeAiRT_Model_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_MODEL)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless MODEL set!")
        return (val,)

class UmeAiRT_VAE_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "Input VAE."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, vae):
        UME_SHARED_STATE[KEY_VAE] = vae
        return ()

class UmeAiRT_VAE_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_VAE)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless VAE set!")
        return (val,)

class UmeAiRT_CLIP_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Input CLIP."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, clip):
        UME_SHARED_STATE[KEY_CLIP] = clip
        return ()

class UmeAiRT_CLIP_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_CLIP)
        if val is None:
            raise ValueError("UmeAiRT: No Wireless CLIP set!")
        return (val,)


# --- LATENT NODES ---

class UmeAiRT_Latent_Input:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Input Latent (Optional override)."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"
    OUTPUT_NODE = True

    def set_val(self, latent):
        UME_SHARED_STATE[KEY_LATENT] = latent
        return ()

class UmeAiRT_Latent_Output:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {} }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "get_val"
    CATEGORY = "UmeAiRT/Wireless/Variables"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def get_val(self):
        val = UME_SHARED_STATE.get(KEY_LATENT)
        if val is None:
             # Return empty dummy if missing to avoid crash, but KSampler checks validation naturally
             raise ValueError("UmeAiRT: No Wireless Latent set.")
        return (val,)


# --- CHECKPOINT NODES ---

class UmeAiRT_WirelessCheckpointLoader(comfy_nodes.CheckpointLoaderSimple):
    """
    Sets the global Wireless Model, CLIP, VAE, and Model Name.
    Resets the Wireless LoRA list to empty on each load to ensure fresh state.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Select a Stable Diffusion checkpoint to load globally."}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint_wireless"
    CATEGORY = "UmeAiRT/Wireless/Loaders"
    OUTPUT_NODE = True 
    
    def load_checkpoint_wireless(self, ckpt_name):
        # Call original loader
        out = super().load_checkpoint(ckpt_name)
        # Set Wireless State
        UME_SHARED_STATE[KEY_MODEL] = out[0]
        UME_SHARED_STATE[KEY_CLIP] = out[1]
        UME_SHARED_STATE[KEY_VAE] = out[2]
        UME_SHARED_STATE[KEY_MODEL_NAME] = ckpt_name
        UME_SHARED_STATE[KEY_LORAS] = []
        log_node(f"Wireless Checkpoint Loaded: {ckpt_name}", color="GREEN")
        return out


# --- LOADER NODES ---

class UmeAiRT_MultiLoraLoader:
    def __init__(self):
        self.lora_loader = comfy_nodes.LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input Model."}),
                "clip": ("CLIP", {"tooltip": "Input CLIP."}),
                
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
        
        # Helper to apply lora if enabled
        loaded_loras = []  # Fresh list per execution (prevents accumulation from prior runs)

        def apply_lora(curr_model, curr_clip, is_on, name, strength):
            if is_on and name != "None":
                # LoraLoader returns (model, clip)
                # Helper to update key
                loaded_loras.append({"name": name, "strength": strength})
                return self.lora_loader.load_lora(curr_model, curr_clip, name, strength, strength)
            return curr_model, curr_clip
        
        # Pipeline
        m, c = apply_lora(model, clip, lora_1, lora_1_name, lora_1_strength)
        m, c = apply_lora(m, c, lora_2, lora_2_name, lora_2_strength)
        m, c = apply_lora(m, c, lora_3, lora_3_name, lora_3_strength)
        
        UME_SHARED_STATE[KEY_LORAS] = loaded_loras

        return (m, c)

# Alias for backward compatibility / init mapping
UmeAiRT_WirelessModelLoader = UmeAiRT_WirelessCheckpointLoader

