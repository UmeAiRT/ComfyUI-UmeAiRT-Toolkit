"""Tests for modules/image_nodes.py — SourceImage, InpaintComposite, ImageSaver nodes."""
import unittest
from unittest.mock import patch, MagicMock

import folder_paths
from modules.image_nodes import (
    UmeAiRT_PipelineImageSaver,
)




class TestPipelineImageSaver(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_PipelineImageSaver.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_instantiation(self):
        node = UmeAiRT_PipelineImageSaver()
        self.assertIsNotNone(node)




if __name__ == "__main__":
    unittest.main()
