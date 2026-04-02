"""Tests for modules/face_nodes.py — Subject Detailer node."""
import unittest

from modules.face_nodes import (
    UmeAiRT_PipelineSubjectDetailer,
)

class TestPipelineSubjectDetailer(unittest.TestCase):
    def test_input_types_required(self):
        inputs = UmeAiRT_PipelineSubjectDetailer.INPUT_TYPES()
        req = inputs["required"]
        self.assertIn("gen_pipe", req)
        self.assertIn("denoise", req)
        self.assertIn("enabled", req)
        self.assertIn("subject", req)
        
        # Test optional overrides
        opt = inputs.get("optional", {})
        # If advanced is merged with required or optional, 
        # actually in face_nodes we put them in required!
        self.assertIn("guide_size", req)
        self.assertIn("max_size", req)

    def test_return_types(self):
        self.assertEqual(UmeAiRT_PipelineSubjectDetailer.RETURN_TYPES, ("UME_PIPELINE",))

    def test_function_name(self):
        self.assertEqual(UmeAiRT_PipelineSubjectDetailer.FUNCTION, "subject_detail")

    def test_category(self):
        self.assertEqual(UmeAiRT_PipelineSubjectDetailer.CATEGORY, "UmeAiRT/Post-Process")

    def test_has_subject_detail_method(self):
        node = UmeAiRT_PipelineSubjectDetailer()
        self.assertTrue(hasattr(node, "subject_detail"))
        self.assertTrue(callable(node.subject_detail))


if __name__ == "__main__":
    unittest.main()
