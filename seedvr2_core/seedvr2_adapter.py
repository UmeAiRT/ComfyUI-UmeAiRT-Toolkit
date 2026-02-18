"""Adapter for calling SeedVR2 VideoUpscaler.

Provides a simple interface to execute SeedVR2 upscaling
using the DiT and VAE configurations from the loader nodes.
Suppresses the verbose forced logs from SeedVR2 (banner, phase
headers, footer, etc.) to keep the console clean.
"""

from __future__ import annotations

import sys
import os
import torch
from typing import Any, Dict
from contextlib import contextmanager

try:
    from logger import log_node
except ImportError:
    def log_node(msg, **kwargs):
        print(f"[UmeAiRT-Toolkit] {msg}")


@contextmanager
def _quiet_seedvr2():
    """Suppress SeedVR2's forced stdout logs (banner, phases, footer).

    stderr is left untouched so real errors still appear.
    The tqdm progress bar writes to stderr so it remains visible too.
    """
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def get_upscaler_class():
    """Get the SeedVR2VideoUpscaler class from ComfyUI's node registry.

    Returns:
        The SeedVR2VideoUpscaler class

    Raises:
        RuntimeError: If SeedVR2VideoUpscaler is not found
    """
    import nodes

    for node_class in nodes.NODE_CLASS_MAPPINGS.values():
        if getattr(node_class, "__name__", "") == "SeedVR2VideoUpscaler":
            return node_class

    raise RuntimeError(
        "❌ SeedVR2VideoUpscaler node not found.\n\n"
        "💡 Solution: Install ComfyUI-SeedVR2_VideoUpscaler v2.5 or later."
    )


def execute_seedvr2(
    *,
    images: torch.Tensor,
    dit_config: Dict[str, Any],
    vae_config: Dict[str, Any],
    seed: int,
    resolution: int,
    batch_size: int = 1,
    color_correction: str = "lab",
) -> torch.Tensor:
    """Execute SeedVR2 upscaling on a batch of images.

    Args:
        images: Input images tensor (N, H, W, C) in [0, 1] range
        dit_config: DiT model configuration from SeedVR2LoadDiTModel node
        vae_config: VAE model configuration from SeedVR2LoadVAEModel node
        seed: Random seed for reproducibility
        resolution: Target resolution for the shortest edge
        batch_size: Number of frames to process together
        color_correction: Color correction method

    Returns:
        Upscaled images tensor (N, H', W', C) in [0, 1] range
    """
    upscaler_cls = get_upscaler_class()

    # Execute SeedVR2 with suppressed verbose logs
    with _quiet_seedvr2():
        result = upscaler_cls.execute(
            image=images,
            dit=dit_config,
            vae=vae_config,
            seed=seed,
            resolution=resolution,
            max_resolution=0,  # No limit
            batch_size=batch_size,
            uniform_batch_size=False,
            temporal_overlap=0,
            prepend_frames=0,
            color_correction=color_correction,
            input_noise_scale=0.0,
            latent_noise_scale=0.0,
            offload_device=dit_config.get("offload_device", "none"),
            enable_debug=False,
        )

    # Extract tensor from io.NodeOutput
    if hasattr(result, "values"):
        tensor = result.values[0] if isinstance(result.values, (list, tuple)) else result.values
    elif hasattr(result, "__getitem__"):
        tensor = result[0]
    else:
        tensor = result

    return tensor
