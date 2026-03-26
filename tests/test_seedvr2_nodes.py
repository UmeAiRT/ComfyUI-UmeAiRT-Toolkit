"""Tests for modules/seedvr2_nodes.py — SeedVR2 Upscale pipeline nodes."""
import unittest

from modules.seedvr2_nodes import (
    UmeAiRT_PipelineSeedVR2Upscale,
    UmeAiRT_PipelineSeedVR2Upscale_Advanced,
)


class TestPipelineSeedVR2Upscale(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_PipelineSeedVR2Upscale.INPUT_TYPES()
        self.assertIn("required", inputs)
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("enabled", req)
        self.assertIn("model", req)
        self.assertIn("upscale_by", req)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.FUNCTION, "upscale")

    def test_category(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.CATEGORY, "UmeAiRT/Pipeline/Post-Processing")

    def test_has_upscale_method(self):
        node = UmeAiRT_PipelineSeedVR2Upscale()
        self.assertTrue(callable(node.upscale))

    def test_has_build_configs(self):
        self.assertTrue(hasattr(UmeAiRT_PipelineSeedVR2Upscale, "_build_configs"))
        self.assertTrue(callable(UmeAiRT_PipelineSeedVR2Upscale._build_configs))


class TestPipelineSeedVR2UpscaleAdvanced(unittest.TestCase):
    def test_input_types_has_tiling_params(self):
        inputs = UmeAiRT_PipelineSeedVR2Upscale_Advanced.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("tile_width", req)
        self.assertIn("tile_height", req)
        self.assertIn("tiling_strategy", req)
        self.assertIn("blending_method", req)
        self.assertIn("color_correction", req)

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale_Advanced.FUNCTION, "upscale")

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale_Advanced.RETURN_TYPES, ("UME_PIPELINE",))


if __name__ == "__main__":
    unittest.main()
