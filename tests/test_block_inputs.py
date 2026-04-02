"""
UmeAiRT Toolkit - Unit Tests for Block Inputs
-----------------------------------------------
Tests for LoRA factory pattern and ControlNet node definitions.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Force UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mock ComfyUI
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['nodes'] = MagicMock()
mock_fp = MagicMock()
mock_fp.get_filename_list.return_value = ["test_lora.safetensors", "style.safetensors"]
# sys.modules['folder_paths'] = mock_fp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.block_inputs import (
    UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3,
    UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10,
    get_lora_inputs, process_lora_stack,
    UmeAiRT_ImageProcess_Img2Img,
    UmeAiRT_ImageProcess_Inpaint,
    UmeAiRT_ImageProcess_Outpaint
)


class TestLoraFactory(unittest.TestCase):
    """Tests for the LoRA Block factory pattern."""

    def test_all_four_classes_exist(self):
        """All 4 LoRA block classes should be created by the factory."""
        for cls in [UmeAiRT_LoraBlock_1, UmeAiRT_LoraBlock_3,
                    UmeAiRT_LoraBlock_5, UmeAiRT_LoraBlock_10]:
            self.assertTrue(hasattr(cls, 'INPUT_TYPES'))
            self.assertTrue(hasattr(cls, 'FUNCTION'))
            self.assertEqual(cls.FUNCTION, "process")
            self.assertEqual(cls.CATEGORY, "UmeAiRT/Loaders/LoRA")
            self.assertEqual(cls.RETURN_TYPES, ("UME_LORA_STACK",))

    def test_class_names_are_correct(self):
        """Factory-created classes must have proper __name__ for ComfyUI registration."""
        self.assertEqual(UmeAiRT_LoraBlock_1.__name__, "UmeAiRT_LoraBlock_1")
        self.assertEqual(UmeAiRT_LoraBlock_3.__name__, "UmeAiRT_LoraBlock_3")
        self.assertEqual(UmeAiRT_LoraBlock_5.__name__, "UmeAiRT_LoraBlock_5")
        self.assertEqual(UmeAiRT_LoraBlock_10.__name__, "UmeAiRT_LoraBlock_10")

    def test_input_counts_differ(self):
        """Each LoRA block should have the correct number of input slots."""
        inputs_1 = UmeAiRT_LoraBlock_1.INPUT_TYPES()
        inputs_3 = UmeAiRT_LoraBlock_3.INPUT_TYPES()
        inputs_10 = UmeAiRT_LoraBlock_10.INPUT_TYPES()

        # Count lora_N_name keys in optional
        names_1 = [k for k in inputs_1["optional"] if k.endswith("_name")]
        names_3 = [k for k in inputs_3["optional"] if k.endswith("_name")]
        names_10 = [k for k in inputs_10["optional"] if k.endswith("_name")]

        self.assertEqual(len(names_1), 1)
        self.assertEqual(len(names_3), 3)
        self.assertEqual(len(names_10), 10)


class TestGetLoraInputs(unittest.TestCase):
    """Tests for the get_lora_inputs helper."""

    def test_generates_correct_keys(self):
        inputs = get_lora_inputs(3)
        self.assertIn("loras", inputs["optional"])
        for i in range(1, 4):
            self.assertIn(f"lora_{i}_on", inputs["optional"])
            self.assertIn(f"lora_{i}_name", inputs["optional"])
            self.assertIn(f"lora_{i}_strength", inputs["optional"])

    def test_all_inputs_have_tooltips(self):
        """Every input parameter should have a tooltip."""
        inputs = get_lora_inputs(3)
        for section in ("required", "optional"):
            for key, spec in inputs.get(section, {}).items():
                if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[1], dict):
                    self.assertIn("tooltip", spec[1],
                                  f"Missing tooltip for '{key}' in get_lora_inputs")


class TestProcessLoraStack(unittest.TestCase):
    """Tests for the process_lora_stack function."""

    def test_empty_produces_empty_stack(self):
        result = process_lora_stack(None)
        self.assertEqual(result, ([],))

    def test_chaining_extends_stack(self):
        existing = [("existing_lora", 1.0, 1.0)]
        result = process_lora_stack(
            existing,
            lora_1_on=True,
            lora_1_name="new_lora.safetensors",
            lora_1_strength=0.8
        )
        stack = result[0]
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack[0][0], "existing_lora")
        self.assertEqual(stack[1][0], "new_lora.safetensors")
        self.assertAlmostEqual(stack[1][1], 0.8)

    def test_disabled_lora_excluded(self):
        result = process_lora_stack(
            None,
            lora_1_on=False,
            lora_1_name="disabled_lora.safetensors",
            lora_1_strength=1.0
        )
        self.assertEqual(result, ([],))

    def test_none_name_excluded(self):
        result = process_lora_stack(
            None,
            lora_1_on=True,
            lora_1_name="None",
            lora_1_strength=1.0
        )
        self.assertEqual(result, ([],))


class TestImageProcessNodes(unittest.TestCase):
    def test_img2img_node(self):
        inputs = UmeAiRT_ImageProcess_Img2Img.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("denoise", inputs["required"])
        self.assertNotIn("mask_blur", inputs.get("optional", {}))
        self.assertNotIn("padding_left", inputs["required"])

    def test_inpaint_node(self):
        inputs = UmeAiRT_ImageProcess_Inpaint.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("denoise", inputs["required"])
        self.assertIn("mask_blur", inputs.get("optional", {}))
        self.assertNotIn("padding_left", inputs["required"])

    def test_outpaint_node(self):
        inputs = UmeAiRT_ImageProcess_Outpaint.INPUT_TYPES()
        self.assertIn("required", inputs)
        self.assertIn("denoise", inputs["required"])
        self.assertIn("target_width", inputs["required"])
        self.assertIn("target_height", inputs["required"])
        self.assertIn("mask_blur", inputs.get("optional", {}))
        self.assertNotIn("padding_left", inputs["required"])


if __name__ == "__main__":
    unittest.main()
