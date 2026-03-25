"""Tests for modules/logic_nodes.py — pipeline upscale, detailer, and base nodes."""
import unittest
from unittest.mock import patch, MagicMock

from modules.logic_nodes import (
    UmeAiRT_PipelineSeedVR2Upscale,
    UmeAiRT_PipelineFaceDetailer,
    UmeAiRT_UltimateUpscale_Base,
)


class TestPipelineSeedVR2Upscale(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_PipelineSeedVR2Upscale.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_has_upscale_method(self):
        node = UmeAiRT_PipelineSeedVR2Upscale()
        self.assertTrue(hasattr(node, "upscale"))
        self.assertTrue(callable(node.upscale))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineSeedVR2Upscale.FUNCTION, "upscale")


class TestPipelineFaceDetailer(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_PipelineFaceDetailer.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_has_face_detail_method(self):
        node = UmeAiRT_PipelineFaceDetailer()
        self.assertTrue(hasattr(node, "face_detail"))
        self.assertTrue(callable(node.face_detail))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineFaceDetailer.FUNCTION, "face_detail")


class TestUltimateUpscaleBase(unittest.TestCase):
    def test_class_exists(self):
        self.assertIsNotNone(UmeAiRT_UltimateUpscale_Base)

    def test_has_get_usdu_node(self):
        self.assertTrue(hasattr(UmeAiRT_UltimateUpscale_Base, "get_usdu_node"))


if __name__ == "__main__":
    unittest.main()
