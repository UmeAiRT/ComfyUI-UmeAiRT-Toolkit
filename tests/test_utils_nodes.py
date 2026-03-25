"""Tests for modules/utils_nodes.py — pack/unpack, label, log, health, signature nodes."""
import unittest
from unittest.mock import patch, MagicMock

from modules.utils_nodes import (
    UmeAiRT_Pack_Bundle,
    UmeAiRT_Unpack_Settings,
    UmeAiRT_Unpack_FilesBundle,
    UmeAiRT_Unpack_ImageBundle,
    UmeAiRT_Unpack_Pipeline,
    UmeAiRT_Unpack_Prompt,
    UmeAiRT_Label,
    UmeAiRT_Bundle_Downloader,
    UmeAiRT_Log_Viewer,
    UmeAiRT_Signature,
    UmeAiRT_HealthCheck,
)
from modules.common import UmeBundle, UmeSettings, UmeImage, GenerationContext


class TestPackBundle(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_Pack_Bundle.INPUT_TYPES()
        self.assertIn("required", inputs)

    def test_pack(self):
        node = UmeAiRT_Pack_Bundle()
        result = node.pack(MagicMock(), MagicMock(), MagicMock(), "test_model")
        self.assertIsInstance(result, tuple)
        bundle = result[0]
        self.assertIsInstance(bundle, UmeBundle)
        self.assertEqual(bundle.model_name, "test_model")


class TestUnpackSettings(unittest.TestCase):
    def test_unpack(self):
        node = UmeAiRT_Unpack_Settings()
        settings = UmeSettings(width=512, height=768, steps=20, cfg=7.0,
                               sampler_name="euler", scheduler="normal", seed=42)
        result = node.unpack(settings)
        self.assertIsInstance(result, tuple)
        # First result should be width
        self.assertEqual(result[0], 512)


class TestUnpackFilesBundle(unittest.TestCase):
    def test_unpack(self):
        node = UmeAiRT_Unpack_FilesBundle()
        bundle = UmeBundle(
            model=MagicMock(), clip=MagicMock(), vae=MagicMock(),
            model_name="test_ckpt"
        )
        result = node.unpack(bundle)
        self.assertIsInstance(result, tuple)


class TestLabel(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_Label.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestBundleDownloader(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_Bundle_Downloader.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestLogViewer(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_Log_Viewer.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestSignature(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_Signature.INPUT_TYPES()
        self.assertIn("required", inputs)


class TestHealthCheck(unittest.TestCase):
    def test_input_types(self):
        inputs = UmeAiRT_HealthCheck.INPUT_TYPES()
        self.assertIn("required", inputs)


if __name__ == "__main__":
    unittest.main()
