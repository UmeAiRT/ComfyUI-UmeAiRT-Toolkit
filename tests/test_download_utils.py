"""Tests for modules/download_utils.py — download, hash, and token utilities."""
import unittest
from unittest.mock import patch, MagicMock

from modules.download_utils import (
    get_hf_token,
    verify_file_hash,
)


class TestGetHfToken(unittest.TestCase):
    @patch.dict("os.environ", {"HF_TOKEN": "test_token_123"})
    def test_reads_from_env(self):
        self.assertEqual(get_hf_token(), "test_token_123")

    @patch.dict("os.environ", {}, clear=True)
    @patch("modules.download_utils.os.path.isfile", return_value=False)
    def test_returns_empty_when_missing(self, mock_isfile):
        self.assertEqual(get_hf_token(), "")

    @patch.dict("os.environ", {"HF_TOKEN": "  spaces  "})
    def test_strips_whitespace(self):
        self.assertEqual(get_hf_token(), "spaces")


class TestVerifyFileHash(unittest.TestCase):
    def test_skip_when_no_expected_hash(self):
        self.assertTrue(verify_file_hash("/any/path", ""))
        self.assertTrue(verify_file_hash("/any/path", None))

    @patch("builtins.open", unittest.mock.mock_open(read_data=b"test data"))
    @patch("modules.download_utils.os.path.basename", return_value="file.bin")
    def test_hash_mismatch_returns_false(self, mock_basename):
        result = verify_file_hash("/fake/file.bin", "0000wrong")
        self.assertFalse(result)

    @patch("builtins.open", side_effect=FileNotFoundError("no file"))
    @patch("modules.download_utils.os.path.basename", return_value="file.bin")
    def test_missing_file_returns_false(self, mock_basename, mock_open):
        result = verify_file_hash("/missing/file.bin", "abc123")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
