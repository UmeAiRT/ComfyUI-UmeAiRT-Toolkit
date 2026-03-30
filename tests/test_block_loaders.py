"""Tests for modules/block_loaders.py — FilesSettings* and BundleLoader nodes."""
import unittest
from unittest.mock import patch, MagicMock

from modules.block_loaders import (
    UmeAiRT_FilesSettings_Checkpoint,
    UmeAiRT_FilesSettings_FLUX,
    UmeAiRT_BundleLoader,
)
from modules.common import UmeBundle


class TestFilesSettingsCheckpoint(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_FilesSettings_Checkpoint.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("optional", inputs)
        self.assertIn("ckpt_name", inputs["required"])
        self.assertIn("vae_name", inputs["optional"])
        self.assertIn("clip_skip", inputs["optional"])

    def test_instantiation(self):
        node = UmeAiRT_FilesSettings_Checkpoint()
        self.assertIsNotNone(node)


class TestFilesSettingsFLUX(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_FilesSettings_FLUX.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestBundleLoader(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_BundleLoader.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_instantiation(self):
        node = UmeAiRT_BundleLoader()
        self.assertIsNotNone(node)


if __name__ == "__main__":
    unittest.main()
