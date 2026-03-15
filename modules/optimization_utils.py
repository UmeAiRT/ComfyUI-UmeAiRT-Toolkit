
import importlib.util
import torch
import sys
import os
import contextlib
from .logger import log_node, GREEN, RED, YELLOW, RESET, CYAN, MAGENTA

# Global Cache
_LIB_CACHE = {}

def check_library(name):
    """Checks if a specific Python library module is installed and caches the result.

    Args:
        name (str): The module name to check (e.g., "sageattention", "triton").

    Returns:
        bool: True if the module is found in the current environment, False otherwise.
    """
    if name in _LIB_CACHE:
        return _LIB_CACHE[name]
    
    spec = importlib.util.find_spec(name)
    found = spec is not None
    _LIB_CACHE[name] = found
    return found

def get_cuda_memory():
    """Retrieves the current CUDA memory usage.

    Returns:
        str: A formatted string showing free and total VRAM in Gigabytes (e.g., '10.50GB free / 24.00GB total').
             Returns "CUDA not available" if torch.cuda is not initialized, or "Unknown" upon failure.
    """
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            return f"{free_gb:.2f}GB free / {total_gb:.2f}GB total"
        return "CUDA not available"
    except Exception:
        return "Unknown"

def check_optimizations():
    """Checks for optimization libraries and system status during startup.

    Logs the presence of crucial performance libraries like SageAttention, Flash Attention, 
    and Triton. It also prints out the initial CUDA memory state to inform the user.
    """
    
    # 1. Library Checks
    sage_status = f"{GREEN}✅{RESET}" if check_library("sageattention") else f"{RED}❌{RESET}"
    flash_status = f"{GREEN}✅{RESET}" if check_library("flash_attn") else f"{RED}❌{RESET}"
    triton_status = f"{GREEN}✅{RESET}" if check_library("triton") else f"{RED}❌{RESET}"
    
    # Using print directly for specific formatting needs or log_node if we want the prefix
    # The user request showed specific icon prefixes. usage of log_node adds [UmeAiRT-Toolkit] prefix.
    # We will use print for the custom lines to match the requested look exactly, 
    # but maybe keep consistent with the project style which uses print in __init__.py too.
    
    log_node(f"⚡ Optimisation check: SageAttention {sage_status} | Flash Attention {flash_status} | Triton {triton_status}")
    
    if not check_library("flash_attn"):
        log_node(f"💡 Optional: pip install flash-attn", color="YELLOW")

    # 3. CUDA Memory
    mem_str = get_cuda_memory()
    log_node(f"📊 Initial CUDA memory: {mem_str}")

    # Return summary for HealthCheck usage
    sage = "✅" if check_library("sageattention") else "❌"
    flash = "✅" if check_library("flash_attn") else "❌"
    triton = "✅" if check_library("triton") else "❌"
    return f"SageAttn={sage} | Flash={flash} | Triton={triton}"


# --- Target-Resolution VAE Warmup ---
_WARMED_UP_SHAPES = set()
_VAE_DECODE_NODE = None

def _get_vae_decode():
    """Returns a singleton VAEDecode node instance to avoid creating disposable objects."""
    global _VAE_DECODE_NODE
    if _VAE_DECODE_NODE is None:
        import nodes
        _VAE_DECODE_NODE = nodes.VAEDecode()
    return _VAE_DECODE_NODE

def warmup_vae(vae, latent_image):
    """Forces an initial VAE compilation pass using an empty tensor of the accurate target resolution.

    This prevents JIT compilers (like Triton or SageAttention) from stalling heavily or spiking VRAM 
    during the first generated frame. By pre-compiling the exact (C, H, W) kernel sizes, 
    subsequent generations are fluid and out-of-memory spikes are avoided.

    Args:
        vae (comfy.sd.VAE): The loaded VAE instance from ComfyUI.
        latent_image (dict): A dictionary containing at least a "samples" key with the target latent shape.
    """
    if not isinstance(latent_image, dict) or "samples" not in latent_image:
        return
        
    if not check_library("triton"):
        return # No Triton = No heavy JIT stall

    try:
        # Extract the exact shape (B, C, H, W) of the target image
        shape = tuple(latent_image["samples"].shape)
        if len(shape) != 4: return
        B, C, H, W = shape
        
        # If we already compiled this exact resolution this session, skip
        if (H, W) in _WARMED_UP_SHAPES:
            return
            
        log_node(f"⚡ Optimisation: VAE Target-Resolution Warmup {H*8}x{W*8} (Preventing VRAM spikes)...", color="CYAN")
        
        vae_decode = _get_vae_decode()
        target_device = latent_image["samples"].device
        
        try:
            # Attempt 1: 16 Channels (FLUX)
            empty_16 = torch.zeros([B, 16, H, W], device=target_device)
            vae_decode.decode(vae, {"samples": empty_16})
        except Exception as e:
            if "channels" in str(e).lower() or "size" in str(e).lower() or "dimension" in str(e).lower():
                # Attempt 2: 4 Channels (SD1.5 / SDXL)
                empty_4 = torch.zeros([B, 4, H, W], device=target_device)
                vae_decode.decode(vae, {"samples": empty_4})
            else:
                raise e
        
        # Remember this shape so we don't compile it again
        _WARMED_UP_SHAPES.add((H, W))
        
    except Exception as e:
        log_node(f"VAE Warmup failed (Safe to ignore): {e}", color="YELLOW")

# --- Sampler Context (Optimizations) ---

class SamplerContext:
    """Context manager that logs active optimizations during sampling.
    
    Note: SageAttention activation should be handled at ComfyUI startup level
    (e.g., via --use-sage-attention CLI flag) where it is properly managed,
    not via per-call global monkey-patching which is thread-unsafe.
    """
    def __init__(self):
        self.optimization_name = "None"
        
    def __enter__(self):
        if check_library("sageattention"):
            self.optimization_name = "SageAttention"
            log_node("⚡ Optimisation Active: SageAttention", color="MAGENTA")
        elif check_library("triton"):
            self.optimization_name = "Triton"
            log_node("⚡ Optimisation Active: Triton", color="MAGENTA")
        else:
            self.optimization_name = "Default"
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No globals to restore — optimizations managed at environment level
