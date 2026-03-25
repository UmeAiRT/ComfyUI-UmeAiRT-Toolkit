"""Tests for modules/image_nodes.py — SourceImage, InpaintComposite, ImageSaver nodes."""
import unittest
from unittest.mock import patch, MagicMock

import folder_paths
from modules.image_nodes import (
    UmeAiRT_SourceImage_Output,
    UmeAiRT_PipelineInpaintComposite,
    UmeAiRT_PipelineImageSaver,
)


class TestSourceImageOutput(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_SourceImage_Output.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_instantiation(self):
        node = UmeAiRT_SourceImage_Output()
        self.assertIsNotNone(node)


class TestPipelineImageSaver(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_PipelineImageSaver.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_instantiation(self):
        node = UmeAiRT_PipelineImageSaver()
        self.assertIsNotNone(node)


class TestPipelineInpaintComposite(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_PipelineInpaintComposite.INPUT_TYPES()
        self.assertIn("required", inputs)


if __name__ == "__main__":
    unittest.main()
