"""Tests for modules/face_nodes.py — FaceDetailer and BboxDetector nodes."""
import unittest

from modules.face_nodes import (
    UmeAiRT_BboxDetectorLoader,
    UmeAiRT_PipelineFaceDetailer,
)


class TestBboxDetectorLoader(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_BboxDetectorLoader.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("model_name", inputs["required"])

    def test_return_types(self):
        self.assertEqual(UmeAiRT_BboxDetectorLoader.RETURN_TYPES, ("BBOX_DETECTOR",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_BboxDetectorLoader.FUNCTION, "load_bbox")

    def test_category(self):
        self.assertEqual(UmeAiRT_BboxDetectorLoader.CATEGORY, "UmeAiRT/Block/Loaders")


class TestPipelineFaceDetailer(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_PipelineFaceDetailer.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("bbox_detector", req)
        self.assertIn("denoise", req)
        self.assertIn("enabled", req)
        self.assertIn("guide_size", req)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineFaceDetailer.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineFaceDetailer.FUNCTION, "face_detail")

    def test_category(self):
        self.assertEqual(UmeAiRT_PipelineFaceDetailer.CATEGORY, "UmeAiRT/Pipeline/Post-Processing")

    def test_has_face_detail_method(self):
        node = UmeAiRT_PipelineFaceDetailer()
        self.assertTrue(callable(node.face_detail))


if __name__ == "__main__":
    unittest.main()
