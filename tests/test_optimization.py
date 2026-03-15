"""
UmeAiRT Toolkit - Unit Tests for Optimization Utilities
---------------------------------------------------------
Tests for SamplerContext, warmup_vae helpers, and check_library.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Force UTF-8 encoding for headless environments
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mock ComfyUI environment before any imports
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from modules.optimization_utils import (
    check_library, SamplerContext, _get_vae_decode, _LIB_CACHE
)


class TestCheckLibrary(unittest.TestCase):
    """Tests for the library detection cache."""

    def setUp(self):
        _LIB_CACHE.clear()

    def test_existing_module_returns_true(self):
        self.assertTrue(check_library("os"))

    def test_nonexistent_module_returns_false(self):
        self.assertFalse(check_library("this_module_does_not_exist_42"))

    def test_result_is_cached(self):
        check_library("os")
        self.assertIn("os", _LIB_CACHE)
        self.assertTrue(_LIB_CACHE["os"])

    def test_cache_prevents_repeated_lookups(self):
        # Pre-populate cache with a fake result
        _LIB_CACHE["fake_module"] = True
        self.assertTrue(check_library("fake_module"))  # Uses cached value


class TestSamplerContext(unittest.TestCase):
    """Tests for SamplerContext (no monkey-patching)."""

    def test_context_manager_enters_and_exits(self):
        with SamplerContext() as ctx:
            self.assertIsNotNone(ctx)
            self.assertIsInstance(ctx.optimization_name, str)

    def test_no_sdpa_mutation(self):
        """SamplerContext must NOT modify torch.nn.functional.scaled_dot_product_attention."""
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        with SamplerContext():
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original_sdpa)

    def test_optimization_name_is_set(self):
        with SamplerContext() as ctx:
            self.assertIn(ctx.optimization_name, ["SageAttention", "Triton", "Default"])


class TestVAEDecodeSingleton(unittest.TestCase):
    """Tests for the singleton VAEDecode helper."""

    def test_singleton_returns_same_instance(self):
        # Since nodes.VAEDecode is mocked, calling _get_vae_decode twice should
        # return the same object
        import modules.optimization_utils as ou
        ou._VAE_DECODE_NODE = None  # Reset
        first = _get_vae_decode()
        second = _get_vae_decode()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
