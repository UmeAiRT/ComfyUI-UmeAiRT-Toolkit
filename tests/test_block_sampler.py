"""Tests for modules/block_sampler.py — UmeAiRT_BlockSampler node."""
import unittest
from unittest.mock import patch, MagicMock

from modules.block_sampler import UmeAiRT_BlockSampler
from modules.common import GenerationContext, UmeBundle, UmeSettings, UmeImage


class TestBlockSampler(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_BlockSampler.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("model_bundle", inputs["required"])
        self.assertIn("settings", inputs["required"])

    def test_return_types(self):
        self.assertEqual(UmeAiRT_BlockSampler.RETURN_TYPES, ("UME_PIPELINE",))
        self.assertEqual(UmeAiRT_BlockSampler.FUNCTION, "process")

    def test_instantiation(self):
        node = UmeAiRT_BlockSampler()
        self.assertIsNotNone(node)
        self.assertIsNotNone(node._ksampler)
        self.assertIsNotNone(node._vae_decode)


if __name__ == "__main__":
    unittest.main()
