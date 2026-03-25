import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import json
from pathlib import Path

# folder_paths is securely mocked by run_tests.py

from modules.image_saver_core.utils_civitai import (
    civitai_embedding_key_name,
    civitai_lora_key_name,
    get_civitai_sampler_name,
    get_civitai_metadata,
    get_civitai_info,
    download_model_info,
    save_civitai_info_file,
    get_manual_folder,
    get_manual_list,
    append_manual_list
)

class TestCivitaiUtils(unittest.TestCase):

    def test_civitai_keys(self):
        self.assertEqual(civitai_embedding_key_name("test"), "embed:test")
        self.assertEqual(civitai_lora_key_name("test"), "LORA:test")

    def test_get_civitai_sampler_name(self):
        self.assertEqual(get_civitai_sampler_name("euler", "normal"), "Euler")
        self.assertEqual(get_civitai_sampler_name("euler_ancestral", "karras"), "Euler a Karras")
        self.assertEqual(get_civitai_sampler_name("lms", "exponential"), "LMS Exponential")
        self.assertEqual(get_civitai_sampler_name("unknown", "normal"), "unknown")
        self.assertEqual(get_civitai_sampler_name("unknown", "sgm_uniform"), "unknown_sgm_uniform")

    @patch('modules.image_saver_core.utils_civitai.get_civitai_info')
    def test_get_civitai_metadata_no_download(self, mock_info):
        loras = {"LORA:A": ("pathA", 0.5, "hashA")}
        embeddings = {"embed:B": ("pathB", 1.0, "hashB")}
        manual = {"manual": (None, None, "hashC")}
        
        resources, hashes, model_hash = get_civitai_metadata("model.ckpt", "path/model.ckpt", "modelhash", loras, embeddings, manual, False)
        
        self.assertEqual(len(resources), 0)
        self.assertEqual(model_hash, "modelhash")
        self.assertIn("LORA:A", hashes)
        self.assertIn("embed:B", hashes)
        self.assertIn("manual", hashes)
        self.assertIn("model", hashes)
        mock_info.assert_not_called()

    @patch('modules.image_saver_core.utils_civitai.get_civitai_info')
    def test_get_civitai_metadata_with_download(self, mock_info):
        mock_info.return_value = {
            "model": {"name": "TestModel"},
            "name": "v1.0",
            "air": "urn:air:test",
            "id": 123
        }
        
        loras = {"LORA:A": ("pathA", 0.5, "hashA")}
        embeddings = {}
        manual = {}
        
        resources, hashes, model_hash = get_civitai_metadata("model.ckpt", "path/model.ckpt", "modelhash", loras, embeddings, manual, True)
        
        self.assertEqual(len(resources), 2) # model + lora
        self.assertEqual(resources[0]["modelName"], "TestModel")
        self.assertEqual(resources[0]["air"], "urn:air:test")
        self.assertEqual(resources[1]["weight"], 0.5)
        self.assertIsNone(model_hash)

    @patch('modules.image_saver_core.utils_civitai.get_civitai_info')
    def test_get_civitai_metadata_download_fallback(self, mock_info):
        # When info is not found, it populates hashes fallback
        mock_info.return_value = None
        loras = {"LORA:A": ("pathA", 0.5, "hashA")}
        
        resources, hashes, model_hash = get_civitai_metadata("model.ckpt", "path/model.ckpt", "modelhash", loras, {}, {}, True)
        
        self.assertEqual(len(resources), 0)
        self.assertEqual(model_hash, "MODELHASH")
        self.assertIn("LORA:A", hashes)
        self.assertEqual(hashes["LORA:A"], "HASHA")

    @patch('modules.image_saver_core.utils_civitai.download_model_info')
    @patch('modules.image_saver_core.utils_civitai.get_manual_list')
    @patch('modules.image_saver_core.utils_civitai.save_civitai_info_file')
    def test_get_civitai_info_manual(self, mock_save, mock_list, mock_download):
        # Empty hash
        self.assertIsNone(get_civitai_info("path", ""))
        
        # None path (implies manual hash lookup) - not in cache -> triggers download
        mock_list.return_value = {}
        mock_download.return_value = {
            "files": [{"name": "test.safetensors", "hashes": {"SHA256": "hash123"}}],
            "model": {"type": "Checkpoint"}
        }
        
        with patch('modules.image_saver_core.utils_civitai.append_manual_list') as mock_append:
            info = get_civitai_info(None, "hash123")
            self.assertIsNotNone(info)
            mock_download.assert_called_once_with(None, "hash123")
            mock_append.assert_called_once()
            mock_save.assert_called_once()

    @patch('modules.image_saver_core.utils_civitai.download_model_info')
    @patch('modules.image_saver_core.utils_civitai.get_manual_list')
    def test_get_civitai_info_manual_cached(self, mock_list, mock_download):
        # Path is None, but Hash is in cache!
        mock_list.return_value = {"HASH123": {"filename": "test.safetensors", "type": "Checkpoint"}}
        
        with patch("builtins.open", mock_open(read_data='{"cached": "yes"}')):
            info = get_civitai_info(None, "hash123")
            self.assertEqual(info, {"cached": "yes"})
            mock_download.assert_not_called()

    def test_get_civitai_info_file_read(self):
        # Path provided, file exists
        with patch("builtins.open", mock_open(read_data='{"file": "exists"}')):
            info = get_civitai_info("some_path.safetensors", "hash123")
            self.assertEqual(info, {"file": "exists"})

    @patch('modules.image_saver_core.utils_civitai.download_model_info')
    def test_get_civitai_info_file_not_found(self, mock_download):
        mock_download.return_value = {"downloaded": "yes"}
        with patch("builtins.open", side_effect=FileNotFoundError):
            info = get_civitai_info("some_path.safetensors", "hash123")
            self.assertEqual(info, {"downloaded": "yes"})
            mock_download.assert_called_once()

    @patch('modules.image_saver_core.utils_civitai.http_get_json')
    @patch('modules.image_saver_core.utils_civitai.save_civitai_info_file')
    def test_download_model_info(self, mock_save, mock_http):
        mock_http.side_effect = [
            {"modelId": 999, "model": {"name": "test"}}, # Version response
            {"creator": {"username": "foo"}, "description": "desc"} # Model response
        ]
        
        content = download_model_info("test.safetensors", "hash123")
        self.assertIsNotNone(content)
        self.assertEqual(content["creator"]["username"], "foo")
        self.assertEqual(content["model"]["description"], "desc")
        mock_save.assert_called_once()
        
    @patch('modules.image_saver_core.utils_civitai.http_get_json')
    def test_download_model_info_fail(self, mock_http):
        mock_http.return_value = None
        self.assertIsNone(download_model_info("test.safetensors", "hash123"))

    def test_save_civitai_info_file(self):
        with patch("builtins.open", mock_open()) as mock_file:
            self.assertTrue(save_civitai_info_file({"k": "v"}, "test.safetensors"))
            mock_file.assert_called_once()
        
        with patch("builtins.open", side_effect=Exception):
            self.assertFalse(save_civitai_info_file({"k": "v"}, "test.safetensors"))

    @patch('pathlib.Path.mkdir')
    def test_manual_list_methods(self, mock_mkdir):
        with patch("builtins.open", mock_open(read_data='{"hash1": {"filename": "f"}}')):
            data = get_manual_list()
            self.assertIn("hash1", data)
            
        with patch("builtins.open", side_effect=FileNotFoundError):
            data = get_manual_list()
            self.assertEqual(data, {})

        with patch("builtins.open", mock_open()) as mock_file:
            with patch('modules.image_saver_core.utils_civitai.get_manual_list', return_value={"existing": "1"}):
                res = append_manual_list("NEW", {"val": 2})
                self.assertIn("existing", res)
                self.assertIn("NEW", res)
                mock_file.assert_called_once()

if __name__ == '__main__':
    unittest.main()
