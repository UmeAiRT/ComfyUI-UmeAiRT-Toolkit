import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

# ComfyUI dependencies are globally mocked by run_tests.py

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from modules.common import (
    UmeBundle, UmeSettings, UmeImage, GenerationContext,
    validate_bundle, extract_pipeline_params,
)

class TestUmeBundleDataclass(unittest.TestCase):
    def test_defaults(self):
        bundle = UmeBundle()
        self.assertIsNone(bundle.model)
        self.assertIsNone(bundle.clip)
        self.assertIsNone(bundle.vae)
        self.assertEqual(bundle.model_name, "")

    def test_construction_with_values(self):
        model, clip, vae = MagicMock(), MagicMock(), MagicMock()
        bundle = UmeBundle(model=model, clip=clip, vae=vae, model_name="test.safetensors")
        self.assertIs(bundle.model, model)
        self.assertIs(bundle.clip, clip)
        self.assertIs(bundle.vae, vae)
        self.assertEqual(bundle.model_name, "test.safetensors")

    def test_asdict_roundtrip(self):
        model, clip, vae = MagicMock(), MagicMock(), MagicMock()
        bundle = UmeBundle(model=model, clip=clip, vae=vae, model_name="test")
        d = asdict(bundle)
        self.assertIn("model", d)
        self.assertIn("clip", d)
        self.assertIn("vae", d)
        self.assertEqual(d["model_name"], "test")


class TestUmeSettingsDataclass(unittest.TestCase):
    def test_defaults(self):
        settings = UmeSettings()
        self.assertEqual(settings.width, 1024)
        self.assertEqual(settings.height, 1024)
        self.assertEqual(settings.steps, 20)
        self.assertAlmostEqual(settings.cfg, 8.0)
        self.assertEqual(settings.sampler_name, "euler")
        self.assertEqual(settings.scheduler, "normal")
        self.assertEqual(settings.seed, 0)

    def test_custom_values(self):
        settings = UmeSettings(width=512, height=768, steps=30, cfg=4.5,
                               sampler_name="dpmpp_2m", scheduler="karras", seed=42)
        self.assertEqual(settings.width, 512)
        self.assertEqual(settings.height, 768)
        self.assertEqual(settings.steps, 30)
        self.assertAlmostEqual(settings.cfg, 4.5)
        self.assertEqual(settings.sampler_name, "dpmpp_2m")
        self.assertEqual(settings.scheduler, "karras")
        self.assertEqual(settings.seed, 42)


class TestUmeImageDataclass(unittest.TestCase):
    def test_defaults(self):
        img = UmeImage()
        self.assertIsNone(img.image)
        self.assertIsNone(img.mask)
        self.assertEqual(img.mode, "img2img")
        self.assertAlmostEqual(img.denoise, 1.0)
        self.assertFalse(img.auto_resize)
        self.assertEqual(img.controlnets, [])

    def test_controlnets_not_shared(self):
        img1 = UmeImage()
        img2 = UmeImage()
        img1.controlnets.append(("cn1", None, 1.0, 0.0, 1.0))
        self.assertEqual(len(img2.controlnets), 0)

    def test_with_tensor(self):
        if type(torch).__name__ in ("MagicMock", "DummyTorch"):
            return
        tensor = torch.rand(1, 512, 512, 3)
        mask = torch.zeros(1, 512, 512)
        img = UmeImage(image=tensor, mask=mask, mode="inpaint", denoise=0.75)
        self.assertEqual(img.image.shape, (1, 512, 512, 3))
        self.assertEqual(img.mode, "inpaint")
        self.assertAlmostEqual(img.denoise, 0.75)


class TestValidateBundle(unittest.TestCase):
    def test_valid_bundle(self):
        bundle = UmeBundle(model=MagicMock(), clip=MagicMock(), vae=MagicMock())
        validate_bundle(bundle, ["model", "clip", "vae"], context="Test")

    def test_missing_attribute(self):
        bundle = UmeBundle(model=MagicMock())  # clip and vae are None
        with self.assertRaises(ValueError) as cm:
            validate_bundle(bundle, ["model", "clip", "vae"], context="Test")
        self.assertIn("clip", str(cm.exception))
        self.assertIn("vae", str(cm.exception))


class TestBundlePackUnpackRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        from modules.utils_nodes import UmeAiRT_Pack_Bundle, UmeAiRT_Unpack_FilesBundle

        model, clip, vae = MagicMock(), MagicMock(), MagicMock()
        packer = UmeAiRT_Pack_Bundle()
        (bundle,) = packer.pack(model, clip, vae, model_name="roundtrip_model")

        self.assertIsInstance(bundle, UmeBundle)
        self.assertIs(bundle.model, model)

        unpacker = UmeAiRT_Unpack_FilesBundle()
        out_model, out_clip, out_vae, out_name = unpacker.unpack(bundle)

        self.assertIs(out_model, model)
        self.assertIs(out_clip, clip)
        self.assertIs(out_vae, vae)
        self.assertEqual(out_name, "roundtrip_model")


class TestSettingsFlow(unittest.TestCase):
    def test_settings_to_context(self):
        settings = UmeSettings(width=768, height=512, steps=30, cfg=5.0,
                               sampler_name="dpmpp_2m_sde", scheduler="karras", seed=12345)

        ctx = GenerationContext()
        ctx.width = settings.width
        ctx.height = settings.height
        ctx.steps = settings.steps
        ctx.cfg = settings.cfg
        ctx.sampler_name = settings.sampler_name
        ctx.scheduler = settings.scheduler
        ctx.seed = settings.seed

        self.assertEqual(ctx.width, 768)
        self.assertEqual(ctx.height, 512)
        self.assertEqual(ctx.steps, 30)
        self.assertAlmostEqual(ctx.cfg, 5.0)
        self.assertEqual(ctx.sampler_name, "dpmpp_2m_sde")
        self.assertEqual(ctx.scheduler, "karras")
        self.assertEqual(ctx.seed, 12345)

    def test_unpack_settings(self):
        from modules.utils_nodes import UmeAiRT_Unpack_Settings

        settings = UmeSettings(width=768, height=512, steps=30, cfg=5.0,
                               sampler_name="dpmpp_2m_sde", scheduler="karras", seed=12345)
        unpacker = UmeAiRT_Unpack_Settings()
        w, h, steps, cfg, sampler, scheduler, seed = unpacker.unpack(settings)

        self.assertEqual(w, 768)
        self.assertEqual(h, 512)
        self.assertEqual(steps, 30)
        self.assertAlmostEqual(cfg, 5.0)
        self.assertEqual(sampler, "dpmpp_2m_sde")
        self.assertEqual(scheduler, "karras")
        self.assertEqual(seed, 12345)


class TestImageBundleFlow(unittest.TestCase):
    def test_unpack_image_bundle(self):
        from modules.utils_nodes import UmeAiRT_Unpack_ImageBundle

        tensor = torch.rand(1, 256, 256, 3)
        mask = torch.zeros(1, 256, 256)
        img_bundle = UmeImage(image=tensor, mask=mask, mode="inpaint",
                              denoise=0.6, auto_resize=True)

        unpacker = UmeAiRT_Unpack_ImageBundle()
        image, out_mask, mode, denoise, auto_resize = unpacker.unpack(img_bundle)

        self.assertEqual(mode, "inpaint")
        self.assertAlmostEqual(denoise, 0.6)
        self.assertTrue(auto_resize)

class TestPipelineContextFlow(unittest.TestCase):
    def test_bundle_to_context_to_unpack(self):
        from modules.utils_nodes import UmeAiRT_Unpack_Pipeline

        model, clip, vae = MagicMock(), MagicMock(), MagicMock()

        bundle = UmeBundle(model=model, clip=clip, vae=vae, model_name="flux-dev.safetensors")
        settings = UmeSettings(width=768, height=512, steps=25, cfg=3.5,
                               sampler_name="euler", scheduler="simple", seed=999)

        ctx = GenerationContext()
        ctx.model = bundle.model
        ctx.clip = bundle.clip
        ctx.vae = bundle.vae
        ctx.model_name = bundle.model_name
        ctx.width = settings.width
        ctx.height = settings.height
        ctx.steps = settings.steps
        ctx.cfg = settings.cfg
        ctx.sampler_name = settings.sampler_name
        ctx.scheduler = settings.scheduler
        ctx.seed = settings.seed
        ctx.positive_prompt = "a beautiful landscape"
        ctx.negative_prompt = "ugly, blurry"
        ctx.image = torch.rand(1, 512, 768, 3)

        self.assertTrue(ctx.is_ready())

        unpacker = UmeAiRT_Unpack_Pipeline()
        result = unpacker.unpack(ctx)
        (out_image, out_model, out_clip, out_vae, out_name,
         out_pos, out_neg, out_w, out_h, out_steps, out_cfg,
         out_sampler, out_scheduler, out_seed, out_denoise) = result

        self.assertIs(out_model, model)
        self.assertIs(out_clip, clip)
        self.assertIs(out_vae, vae)
        self.assertEqual(out_name, "flux-dev.safetensors")
        self.assertEqual(out_pos, "a beautiful landscape")
        self.assertEqual(out_neg, "ugly, blurry")
        self.assertEqual(out_w, 768)
        self.assertEqual(out_h, 512)
        self.assertEqual(out_steps, 25)
        self.assertAlmostEqual(out_cfg, 3.5)
        self.assertEqual(out_sampler, "euler")
        self.assertEqual(out_scheduler, "simple")
        self.assertEqual(out_seed, 999)

if __name__ == "__main__":
    unittest.main()
