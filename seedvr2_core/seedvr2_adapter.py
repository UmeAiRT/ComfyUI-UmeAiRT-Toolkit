"""Adapter for calling SeedVR2 VideoUpscaler.

Provides a simple interface to execute SeedVR2 upscaling
using the DiT and VAE configurations from the loader nodes.
Suppresses the verbose forced logs from SeedVR2 (banner, phase
headers, footer, etc.) to keep the console clean.

Import strategy (prioritized):
  1. Vendored copy shipped with UmeAiRT-Toolkit (seedvr2_core/vendor/)
  2. External ComfyUI custom node (seedvr2_videoupscaler in custom_nodes/)
  3. Already-importable package (pip-installed or sys.path)
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


# ── Path resolution ─────────────────────────────────────────────────
_seedvr2_path_ready = False


def _ensure_seedvr2_path():
    """Add vendored seedvr2_videoupscaler to sys.path if not already importable."""
    global _seedvr2_path_ready
    if _seedvr2_path_ready:
        return

    # 1. Check if already importable (external install or previous path setup)
    try:
        import seedvr2_videoupscaler  # noqa: F401
        _seedvr2_path_ready = True
        return
    except ImportError:
        pass

    # 2. Vendored copy: seedvr2_core/vendor/
    vendor_dir = os.path.join(os.path.dirname(__file__), "vendor")
    seedvr2_vendor = os.path.join(vendor_dir, "seedvr2_videoupscaler")

    if os.path.isdir(seedvr2_vendor):
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        _seedvr2_path_ready = True
        log_node("SeedVR2: Using vendored copy", color="GREEN")
        return

    # 3. Sibling folder in custom_nodes/ (legacy external install)
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    external_dir = os.path.join(custom_nodes_dir, "seedvr2_videoupscaler")

    if os.path.isdir(external_dir):
        if custom_nodes_dir not in sys.path:
            sys.path.insert(0, custom_nodes_dir)
        _seedvr2_path_ready = True
        log_node("SeedVR2: Using external custom node", color="YELLOW")
        return

    raise RuntimeError(
        "❌ SeedVR2 VideoUpscaler not found.\n\n"
        "💡 This should not happen with a standard UmeAiRT-Toolkit installation.\n"
        "   The vendored copy is missing from seedvr2_core/vendor/.\n"
        "   Try re-installing the toolkit or restoring the vendor directory."
    )


# ── Suppression context ─────────────────────────────────────────────
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


# ── Upscaler class import ───────────────────────────────────────────
def get_upscaler_class():
    """Import SeedVR2VideoUpscaler directly from the Python package.

    Returns:
        The SeedVR2VideoUpscaler class

    Raises:
        RuntimeError: If SeedVR2VideoUpscaler cannot be imported
    """
    _ensure_seedvr2_path()

    try:
        from seedvr2_videoupscaler.src.interfaces.video_upscaler import SeedVR2VideoUpscaler
        return SeedVR2VideoUpscaler
    except ImportError as e:
        raise RuntimeError(
            f"❌ Failed to import SeedVR2VideoUpscaler: {e}\n\n"
            "💡 Solution: Re-install ComfyUI-UmeAiRT-Toolkit to restore the vendored SeedVR2 files."
        )


# ── Public API ───────────────────────────────────────────────────────
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
