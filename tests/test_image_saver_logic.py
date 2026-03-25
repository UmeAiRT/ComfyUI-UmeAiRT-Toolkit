import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Mocks are provided globally by run_tests.py

from modules.image_saver_core.logic import (
    parse_checkpoint_name,
    parse_checkpoint_name_without_extension,
    get_timestamp,
    apply_custom_time_format,
    save_json,
    Metadata,
    ImageSaverLogic
)

class TestImageSaverLogic(unittest.TestCase):

    def test_parse_checkpoint_name(self):
        self.assertEqual(parse_checkpoint_name("models/checkpoints/test.safetensors"), "test.safetensors")
        self.assertEqual(parse_checkpoint_name("test_model"), "test_model")

    def test_parse_checkpoint_name_without_extension(self):
        self.assertEqual(parse_checkpoint_name_without_extension("test.safetensors"), "test")
        self.assertEqual(parse_checkpoint_name_without_extension("test.gguf"), "test")
        self.assertEqual(parse_checkpoint_name_without_extension("test.unknown_ext"), "test.unknown_ext")

    @patch('modules.image_saver_core.logic.datetime')
    def test_get_timestamp(self, mock_datetime):
        mock_now = MagicMock()
        mock_now.strftime.return_value = "2024-01-01"
        mock_datetime.now.return_value = mock_now
        self.assertEqual(get_timestamp("%Y-%m-%d"), "2024-01-01")

        mock_now.strftime.side_effect = [Exception("Format error"), "2024-01-01-120000"]
        self.assertEqual(get_timestamp("%invalid"), "2024-01-01-120000")

    @patch('modules.image_saver_core.logic.datetime')
    def test_apply_custom_time_format(self, mock_datetime):
        mock_now = MagicMock()
        mock_now.strftime.return_value = "custom_time"
        mock_datetime.now.return_value = mock_now
        res = apply_custom_time_format("prefix_%time_format<%Y_%m>_suffix")
        self.assertEqual(res, "prefix_custom_time_suffix")

    @patch('builtins.open', new_callable=mock_open)
    @patch('modules.image_saver_core.logic.json.dump')
    def test_save_json(self, mock_json_dump, mock_file):
        # With workflow
        save_json({'workflow': {'k': 'v'}}, "test_file")
        mock_file.assert_called_with("test_file.json", "w")
        mock_json_dump.assert_called_once()

        # Without workflow
        mock_file.reset_mock()
        mock_json_dump.reset_mock()
        save_json({}, "test_file")
        mock_file.assert_called_with("test_file.json", "w")
        mock_json_dump.assert_called_once_with(None, mock_file())

        # With exception
        mock_file.side_effect = Exception("Write error")
        save_json({'workflow': {}}, "test_file") # Should not throw

    def test_replace_placeholders(self):
        txt = "Image_%width_%height_%sampler_name_%model_%steps_%cfg_%seed"
        res = ImageSaverLogic.replace_placeholders(txt, 512, 512, 123, "model.safetensors", 1, "%Y", "euler", 20, 7.0, "normal", 1.0, 1, "custom")
        self.assertTrue("512_512_euler_model.safetensors_20_7.0_123" in res)

    def test_clean_prompt(self):
        mock_extractor = MagicMock()
        mock_extractor.LORA = r"<lora:[^>]+>"
        mock_extractor.EMBEDDING = r"embedding:([^\s]+)"
        
        prompt = "test prompt <lora:abc:1.0> with embedding:fast_neg and BREAK(x)"
        res = ImageSaverLogic.clean_prompt(prompt, mock_extractor)
        self.assertEqual(res, "test prompt  with fast_neg and ")

    @patch('modules.image_saver_core.logic.full_checkpoint_path_for')
    @patch('modules.image_saver_core.logic.get_sha256')
    def test_get_multiple_models(self, mock_sha, mock_path):
        mock_path.return_value = "fake_path"
        mock_sha.return_value = "1234567890abcdef"
        
        model, hashes = ImageSaverLogic.get_multiple_models("model1.ckpt, model2.ckpt", "initial:hash")
        self.assertEqual(model, "model1.ckpt")
        self.assertEqual(hashes, "initial:hash,model2.ckpt:1234567890")

    def test_parse_manual_hashes(self):
        res = ImageSaverLogic.parse_manual_hashes("model:1234567890, unknown:abcdef1234:0.5, 123456abcd", {"existing_hash"}, True)
        self.assertIn("model", res)
        self.assertEqual(res["model"], (None, None, "1234567890"))
        
        self.assertIn("unknown", res)
        self.assertEqual(res["unknown"], (None, 0.5, "abcdef1234"))
        
        self.assertIn("manual1", res)
        self.assertEqual(res["manual1"], (None, None, "123456abcd"))

    @patch('modules.image_saver_core.logic.full_checkpoint_path_for')
    @patch('modules.image_saver_core.logic.get_sha256')
    @patch('modules.image_saver_core.logic.get_civitai_sampler_name')
    @patch('modules.image_saver_core.logic.get_civitai_metadata')
    def test_make_metadata(self, mock_civitai, mock_sampler, mock_sha, mock_path):
        mock_path.return_value = "path"
        mock_sha.return_value = "hash123456"
        mock_sampler.return_value = "Euler a"
        mock_civitai.return_value = ({"civitai": "data"}, {}, "hash123456")

        meta = ImageSaverLogic.make_metadata("model.ckpt", "pos", "neg", 512, 512, 123, 20, 7.0, "euler", "normal", 1.0, 1, "custom", "add:hash", True, True)
        self.assertEqual(meta.modelname, "model.ckpt")
        self.assertEqual(meta.width, 512)
        self.assertTrue("Euler a" in meta.a111_params)
        self.assertTrue("Civitai resources" in meta.a111_params)

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_get_unique_filename(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = ["test_01.png", "test_05.png", "ignore.png"]
        
        # Batch size 1
        res = ImageSaverLogic.get_unique_filename("fake", "test", "png", 1, 0)
        self.assertEqual(res, "test_06")
        
        # Batch size 4, index 2
        res2 = ImageSaverLogic.get_unique_filename("fake", "test", "png", 4, 2)
        self.assertEqual(res2, "test_08")

        mock_exists.return_value = False
        res3 = ImageSaverLogic.get_unique_filename("fake", "test", "png", 1, 0)
        self.assertEqual(res3, "test_00")

    @patch('modules.image_saver_core.logic.save_image')
    @patch('modules.image_saver_core.logic.save_json')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.listdir')
    def test_save_images(self, mock_listdir, mock_makedirs, mock_exists, mock_json, mock_save):
        mock_listdir.return_value = []
        mock_exists.return_value = False
        
        mock_image = MagicMock()
        mock_image.cpu.return_value.numpy.return_value = __import__('numpy').zeros((512, 512, 3))
        
        mock_meta = MagicMock(spec=Metadata)
        mock_meta.width = 512
        mock_meta.height = 512
        mock_meta.seed = 123
        mock_meta.modelname = "model"
        mock_meta.sampler_name = "euler"
        mock_meta.steps = 20
        mock_meta.cfg = 7.0
        mock_meta.scheduler_name = "normal"
        mock_meta.denoise = 1.0
        mock_meta.clip_skip = 1
        mock_meta.custom = "test"
        mock_meta.a111_params = "params"
        
        # Safe path
        res = ImageSaverLogic.save_images([mock_image, mock_image], "output", "png", "subdir", 100, False, True, {}, {}, True, True, 1, "%Y", mock_meta)
        
        self.assertEqual(len(res), 2)
        mock_save.assert_called()
        mock_json.assert_called()
        
        # Traversal payload
        mock_exists.return_value = True
        res_trav = ImageSaverLogic.save_images([mock_image], "output", "png", "../../windows/system32", 100, False, True, {}, {}, False, False, 1, "%Y", mock_meta)
        self.assertTrue("output.png" in res_trav[0])

if __name__ == '__main__':
    unittest.main()
