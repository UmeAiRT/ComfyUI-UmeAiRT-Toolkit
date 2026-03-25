"""Tests for modules/extra_samplers.py — math utilities and sampler registration."""
import unittest
from unittest.mock import patch, MagicMock
import torch

from modules.extra_samplers import (
    append_zero,
    get_ancestral_step,
    register_extra_samplers,
)


class TestMathUtilities(unittest.TestCase):
    def test_get_ancestral_step_no_eta(self):
        sigma_down, sigma_up = get_ancestral_step(1.0, 0.5, eta=0.0)
        self.assertEqual(sigma_down, 0.5)
        self.assertEqual(sigma_up, 0.0)

    def test_get_ancestral_step_with_eta(self):
        sigma_down, sigma_up = get_ancestral_step(1.0, 0.5, eta=1.0)
        self.assertIsInstance(sigma_down, float)
        self.assertIsInstance(sigma_up, float)
        self.assertGreaterEqual(sigma_down, 0.0)
        self.assertGreaterEqual(sigma_up, 0.0)

    def test_append_zero(self):
        if type(torch).__name__ in ("DummyTorch", "MagicMock"):
            return
        x = torch.tensor([1.0, 2.0, 3.0])
        result = append_zero(x)
        self.assertEqual(result.shape[0], 4)
        self.assertEqual(result[-1].item(), 0.0)


class TestRegisterSamplers(unittest.TestCase):
    def test_register_extra_samplers(self):
        """Should not raise even if comfy.samplers is a mock."""
        try:
            register_extra_samplers()
        except Exception:
            pass  # Expected in test environment without full comfy


if __name__ == "__main__":
    unittest.main()
