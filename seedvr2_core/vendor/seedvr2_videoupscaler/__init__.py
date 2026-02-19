"""
SeedVR2 VideoUpscaler — Vendored for UmeAiRT-Toolkit

This is a vendored copy of ComfyUI-SeedVR2_VideoUpscaler.
Only the compatibility shims are run on import.
Node registration is DISABLED to prevent double-registration
when the external node is also installed.

Original: https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
"""

from .src.optimization.compatibility import ensure_triton_compat  # noqa: F401

# NOTE: We intentionally do NOT import comfy_entrypoint / SeedVR2Extension
# to avoid registering duplicate ComfyUI nodes.