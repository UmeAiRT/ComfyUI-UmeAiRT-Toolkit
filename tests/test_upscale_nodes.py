"""Tests for modules/upscale_nodes.py — UltimateSD Upscale pipeline nodes."""
import unittest

from modules.upscale_nodes import (
    UmeAiRT_PipelineUltimateUpscale,
    UmeAiRT_UltimateUpscale_Base,
)


class TestUltimateUpscaleBase(unittest.TestCase):
    def test_class_exists(self):
        self.assertIsNotNone(UmeAiRT_UltimateUpscale_Base)

    def test_has_encode_prompts(self):
        base = UmeAiRT_UltimateUpscale_Base()
        self.assertTrue(hasattr(base, "encode_prompts"))
        self.assertTrue(callable(base.encode_prompts))

    def test_has_get_usdu_node(self):
        self.assertTrue(hasattr(UmeAiRT_UltimateUpscale_Base, "get_usdu_node"))


class TestPipelineUltimateUpscale(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_PipelineUltimateUpscale.INPUT_TYPES()
        self.assertIn("required", inputs)
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("model", req)
        self.assertIn("upscale_by", req)
        self.assertIn("enabled", req)

    def test_input_types_has_optional(self):
        inputs = UmeAiRT_PipelineUltimateUpscale.INPUT_TYPES()
        self.assertIn("optional", inputs)
        opt = inputs["optional"]
        self.assertIn("tile_width", opt)
        self.assertIn("seam_fix_mode", opt)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineUltimateUpscale.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineUltimateUpscale.FUNCTION, "upscale")

    def test_category(self):
        self.assertEqual(UmeAiRT_PipelineUltimateUpscale.CATEGORY, "UmeAiRT/Pipeline/Post-Processing")

    def test_has_upscale_method(self):
        node = UmeAiRT_PipelineUltimateUpscale()
        self.assertTrue(callable(node.upscale))

    def test_inherits_base(self):
        self.assertTrue(issubclass(UmeAiRT_PipelineUltimateUpscale, UmeAiRT_UltimateUpscale_Base))


if __name__ == "__main__":
    unittest.main()
