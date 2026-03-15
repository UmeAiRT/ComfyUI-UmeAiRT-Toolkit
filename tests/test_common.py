"""
UmeAiRT Toolkit - Unit Tests for Common Module
------------------------------------------------
Tests for GenerationContext, resize_tensor, encode_prompts,
and apply_outpaint_padding from modules/common.py.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Force UTF-8 encoding for headless environments
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mock ComfyUI environment before any imports
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['nodes'] = MagicMock()
sys.modules['folder_paths'] = MagicMock()

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from modules.common import GenerationContext, resize_tensor, encode_prompts, apply_outpaint_padding


class TestGenerationContext(unittest.TestCase):
    """Tests for GenerationContext lifecycle and cloning."""

    def test_init_defaults(self):
        ctx = GenerationContext()
        self.assertIsNone(ctx.model)
        self.assertIsNone(ctx.clip)
        self.assertIsNone(ctx.vae)
        self.assertEqual(ctx.model_name, "")
        self.assertEqual(ctx.width, 1024)
        self.assertEqual(ctx.height, 1024)
        self.assertEqual(ctx.steps, 20)
        self.assertAlmostEqual(ctx.cfg, 8.0)
        self.assertEqual(ctx.seed, 0)
        self.assertIsNone(ctx.image)
        self.assertEqual(ctx.loras, [])
        self.assertEqual(ctx.controlnets, [])

    def test_is_ready_false_when_empty(self):
        ctx = GenerationContext()
        self.assertFalse(ctx.is_ready())

    def test_is_ready_true_when_set(self):
        ctx = GenerationContext()
        ctx.model = MagicMock()
        ctx.clip = MagicMock()
        ctx.vae = MagicMock()
        self.assertTrue(ctx.is_ready())

    def test_clone_produces_independent_lists(self):
        ctx = GenerationContext()
        ctx.loras = [("lora1", 1.0, 1.0)]
        ctx.controlnets = [("cn1",)]
        ctx.model = MagicMock()
        ctx.width = 512

        cloned = ctx.clone()

        # Values should match
        self.assertEqual(cloned.width, 512)
        self.assertIs(cloned.model, ctx.model)  # shallow copy shares model ref

        # Lists should be independent copies
        cloned.loras.append(("lora2", 0.5, 0.5))
        self.assertEqual(len(ctx.loras), 1)      # original unchanged
        self.assertEqual(len(cloned.loras), 2)    # cloned has new item

        cloned.controlnets.append(("cn2",))
        self.assertEqual(len(ctx.controlnets), 1)


class TestResizeTensor(unittest.TestCase):
    """Tests for resize_tensor utility."""

    def test_resize_image(self):
        # [B, H, W, C] image tensor
        img = torch.rand(1, 100, 200, 3)
        resized = resize_tensor(img, 50, 100)
        self.assertEqual(resized.shape, (1, 50, 100, 3))

    def test_resize_mask(self):
        # [B, H, W] mask tensor
        mask = torch.rand(1, 100, 200)
        resized = resize_tensor(mask, 50, 100, is_mask=True)
        self.assertEqual(resized.shape, (1, 50, 100))

    def test_resize_preserves_batch_dim(self):
        img = torch.rand(4, 100, 200, 3)
        resized = resize_tensor(img, 50, 50)
        self.assertEqual(resized.shape[0], 4)


class TestEncodePrompts(unittest.TestCase):
    """Tests for encode_prompts utility."""

    def test_returns_correct_format(self):
        mock_clip = MagicMock()
        mock_clip.tokenize.return_value = "tokens"
        mock_cond = torch.rand(1, 77, 768)
        mock_pooled = torch.rand(1, 768)
        mock_clip.encode_from_tokens.return_value = (mock_cond, mock_pooled)

        pos, neg = encode_prompts(mock_clip, "a cat", "bad quality")

        # Each should be [[cond, {"pooled_output": pooled}]]
        self.assertEqual(len(pos), 1)
        self.assertEqual(len(neg), 1)
        self.assertIn("pooled_output", pos[0][1])
        self.assertIn("pooled_output", neg[0][1])
        
        # clip.tokenize should be called twice (pos + neg)
        self.assertEqual(mock_clip.tokenize.call_count, 2)


class TestApplyOutpaintPadding(unittest.TestCase):
    """Tests for apply_outpaint_padding utility."""

    def test_no_padding_returns_originals(self):
        img = torch.rand(1, 100, 100, 3)
        mask = torch.zeros(1, 100, 100)
        result_img, result_mask = apply_outpaint_padding(img, mask, 0, 0, 0, 0)
        self.assertTrue(torch.equal(result_img, img))
        self.assertTrue(torch.equal(result_mask, mask))

    def test_padding_increases_dimensions(self):
        img = torch.rand(1, 100, 100, 3)
        mask = None
        result_img, result_mask = apply_outpaint_padding(img, mask, 10, 20, 10, 20)
        self.assertEqual(result_img.shape, (1, 140, 120, 3))  # H+40, W+20
        self.assertEqual(result_mask.shape, (1, 140, 120))

    def test_padding_with_existing_mask(self):
        img = torch.rand(1, 50, 50, 3)
        mask = torch.ones(1, 50, 50)
        result_img, result_mask = apply_outpaint_padding(img, mask, 5, 5, 5, 5)
        self.assertEqual(result_img.shape, (1, 60, 60, 3))
        self.assertEqual(result_mask.shape, (1, 60, 60))


if __name__ == "__main__":
    unittest.main()
