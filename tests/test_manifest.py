"""Tests for modules/manifest.py — manifest loading and bundle dropdown population."""
import unittest
from unittest.mock import patch, MagicMock

from modules.manifest import (
    load_manifest,
    get_bundle_dropdowns,
    find_file_in_folders,
    get_download_dest,
    _get_manifest_cache_path,
)


class TestGetManifestCachePath(unittest.TestCase):
    def test_returns_path(self):
        path = _get_manifest_cache_path()
        self.assertIn("model_manifest.json", path)
        self.assertIn("umeairt", path)


class TestLoadManifest(unittest.TestCase):
    @patch("modules.manifest._MANIFEST_CACHE", None)
    @patch("modules.manifest.os.path.exists", return_value=False)
    @patch("modules.manifest.urllib.request.urlopen", side_effect=Exception("no network"))
    def test_returns_empty_dict_on_failure(self, mock_urlopen, mock_exists):
        result = load_manifest()
        self.assertIsInstance(result, dict)


class TestGetBundleDropdowns(unittest.TestCase):
    @patch("modules.manifest.load_manifest")
    def test_returns_tuple_of_lists(self, mock_load):
        mock_load.return_value = {}
        categories, versions = get_bundle_dropdowns()
        self.assertIsInstance(categories, list)
        self.assertIsInstance(versions, list)
        self.assertIn("No Bundles Found", categories)

    @patch("modules.manifest.load_manifest")
    def test_parses_legacy_structure(self, mock_load):
        mock_load.return_value = {
            "SDXL": {
                "standard": {"files": []},
                "_meta": {"base_url": "https://example.com"}
            }
        }
        categories, versions = get_bundle_dropdowns()
        self.assertIn("SDXL", categories)
        self.assertIn("standard", versions)


class TestFindFileInFolders(unittest.TestCase):
    @patch("modules.manifest.folder_paths.get_full_path", return_value=None)
    def test_returns_none_when_not_found(self, mock_get):
        result = find_file_in_folders("nonexistent.safetensors", ["checkpoints"])
        self.assertIsNone(result)

    @patch("modules.manifest.os.path.exists", side_effect=lambda p: not p.endswith(".aria2") and not p.endswith(".download"))
    @patch("modules.manifest.folder_paths.get_full_path", return_value="/models/checkpoints/model.safetensors")
    def test_returns_path_when_found(self, mock_get, mock_exists):
        result = find_file_in_folders("model.safetensors", ["checkpoints"])
        self.assertEqual(result, "/models/checkpoints/model.safetensors")


class TestGetDownloadDest(unittest.TestCase):
    @patch("modules.manifest.folder_paths.get_folder_paths", return_value=["/models/checkpoints"])
    @patch("modules.manifest.os.makedirs")
    def test_returns_path(self, mock_makedirs, mock_get_paths):
        result = get_download_dest("model.safetensors", "checkpoints")
        self.assertIn("model.safetensors", result)


if __name__ == "__main__":
    unittest.main()
