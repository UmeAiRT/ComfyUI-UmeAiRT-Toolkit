"""Tests for modules/image_saver_core/utils.py — sanitize, sha256, path resolution."""
import unittest
from unittest.mock import patch, MagicMock

import folder_paths
from modules.image_saver_core.utils import (
    sanitize_filename,
    get_sha256,
    full_checkpoint_path_for,
    full_lora_path_for,
    full_embedding_path_for,
    get_file_path_iterator,
    custom_file_path_generator,
    http_get_json,
)


class TestSanitizeFilename(unittest.TestCase):
    def test_safe_name_unchanged(self):
        self.assertEqual(sanitize_filename("safe_name"), "safe_name")

    def test_unsafe_chars_removed(self):
        result = sanitize_filename('un<sa>fe:"name"')
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)
        self.assertNotIn(":", result)
        self.assertNotIn('"', result)

    def test_trailing_dots_stripped(self):
        self.assertEqual(sanitize_filename("name...  "), "name")


class TestGetSha256(unittest.TestCase):
    @patch("modules.image_saver_core.utils.os.path.exists", return_value=True)
    @patch("builtins.open", unittest.mock.mock_open(read_data="abc123hash"))
    def test_reads_existing_hash_file(self, mock_exists):
        result = get_sha256("model.safetensors")
        self.assertEqual(result, "abc123hash")


class TestFullCheckpointPathFor(unittest.TestCase):
    def test_empty_name_returns_empty(self):
        self.assertEqual(full_checkpoint_path_for(""), "")

    @patch("modules.image_saver_core.utils.get_file_path_match", return_value="model.safetensors")
    @patch.object(folder_paths, "get_full_path", return_value="/models/checkpoints/model.safetensors")
    def test_returns_full_path(self, mock_full_path, mock_match):
        result = full_checkpoint_path_for("model")
        self.assertEqual(result, "/models/checkpoints/model.safetensors")


class TestFullLoraPathFor(unittest.TestCase):
    @patch("modules.image_saver_core.utils.get_file_path_match", return_value=None)
    def test_returns_none_when_not_found(self, mock_match):
        result = full_lora_path_for("nonexistent")
        self.assertIsNone(result)


class TestFullEmbeddingPathFor(unittest.TestCase):
    @patch("modules.image_saver_core.utils.get_file_path_match", return_value=None)
    def test_returns_none_when_not_found(self, mock_match):
        result = full_embedding_path_for("nonexistent")
        self.assertIsNone(result)


class TestHttpGetJson(unittest.TestCase):
    @patch("modules.image_saver_core.utils.requests.get")
    def test_returns_json(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"key": "value"}
        mock_get.return_value = mock_resp
        result = http_get_json("http://example.com/api")
        self.assertEqual(result, {"key": "value"})

    @patch("modules.image_saver_core.utils.requests.get")
    def test_returns_none_on_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"
        mock_get.return_value = mock_resp
        result = http_get_json("http://example.com/missing")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
