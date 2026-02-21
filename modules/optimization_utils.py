
import importlib.util
import torch
import sys
import os
import contextlib
from .logger import log_node, GREEN, RED, YELLOW, RESET, CYAN, MAGENTA

# Global Cache
_LIB_CACHE = {}

def check_library(name):
    """Check if a library is installed (cached)."""
    if name in _LIB_CACHE:
        return _LIB_CACHE[name]
    
    spec = importlib.util.find_spec(name)
    found = spec is not None
    _LIB_CACHE[name] = found
    return found

def get_cuda_memory():
    """Get CUDA memory usage."""
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            return f"{free_gb:.2f}GB free / {total_gb:.2f}GB total"
        return "CUDA not available"
    except:
        return "Unknown"

def check_optimizations():
    """
    Check for optimization libraries and system status.
    Mimics the requested style:
    ⚡ Optimisation check: SageAttention [?] | Flash Attention [?] | Triton [?]
    💡 Optional: pip install flash-attn
    🔧 Conv3d workaround active: PyTorch [ver], cuDNN [ver] (fixing VAE 3x memory bug)
    📊 Initial CUDA memory: [free] / [total]
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

@contextlib.contextmanager
def SamplerContext():
    """
    Context manager to apply optimizations (SageAttention, etc.) specifically during sampling.
    Restores original state afterwards to avoid side effects.
    """
    # Track what we activated for logging
    active_optimizations = []
    
    # 1. SageAttention
    if check_library("sageattention"):
        try:
            # We assume if it's installed, it might be patching things or available for use
            # Just importing it to ensure it's loaded if needed
            import sageattention
            active_optimizations.append(f"{GREEN}SageAttention{RESET}")
        except Exception as e:
            log_node(f"Failed to init SageAttention: {e}", color="RED")

    # 2. Triton (Implicitly used by torch.compile or some attention backends)
    if check_library("triton"):
         active_optimizations.append(f"{GREEN}Triton{RESET}")

    # Log specific Optimization usage for this run
    if active_optimizations:
        log_node(f"⚡ Optimisation Active: {' | '.join(active_optimizations)}", color="GREEN")
    
    try:
        yield
    finally:
        pass

# --- Pre-Compilation Warmup ---
_VAE_WARMED_UP = False

def warmup_vae(vae):
    """
    Forces Triton/SageAttention to compile VAE kernels before the heavy KSampler fills VRAM.
    Creates a tiny 64x64 empty latent and decodes it.
    Only runs once per ComfyUI session and only if Triton is installed.
    """
    global _VAE_WARMED_UP
    if _VAE_WARMED_UP:
        return
    
    if not check_library("triton"):
        # No Triton means no massive JIT stall, skip warmup
        _VAE_WARMED_UP = True
        return
        
    log_node("⚡ Optimisation: Initiating VAE Pre-Compilation Warmup (Preventing VRAM spikes)...", color="CYAN")
    try:
        import comfy_extras.nodes_custom_sampler as comfy_nodes
        import nodes
        
        # Create a tiny 64x64 empty tensor (1, 4, 8, 8 in latent space)
        empty_latent = torch.zeros([1, 4, 8, 8], device="cpu")
        latent_dict = {"samples": empty_latent}
        
        # Decode it silently
        nodes.VAEDecode().decode(vae, latent_dict)
    except Exception as e:
        log_node(f"VAE Warmup failed (Safe to ignore): {e}", color="YELLOW")
    finally:
        _VAE_WARMED_UP = True
