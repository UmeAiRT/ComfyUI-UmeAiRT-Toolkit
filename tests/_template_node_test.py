"""
Template for testing a new ComfyUI node.

Usage:
  1. Copy this file → tests/test_<your_module>.py
  2. Replace TODO placeholders with your actual class names
  3. Run: python run_tests.py

Every node should have at least these structural tests to catch
import errors, missing INPUT_TYPES, and missing FUNCTION methods.
"""
import unittest

# TODO: Replace with your actual import
# from modules.your_module import YourNodeClass


class TestYourNodeClass(unittest.TestCase):
    """Structural tests for YourNodeClass."""

    # TODO: Uncomment and replace YourNodeClass

    # def test_input_types_exists(self):
    #     """INPUT_TYPES must return a dict with 'required' key."""
    #     inputs = YourNodeClass.INPUT_TYPES()
    #     self.assertIsInstance(inputs, dict)
    #     self.assertIn("required", inputs)

    # def test_function_method_exists(self):
    #     """The method referenced by FUNCTION must exist and be callable."""
    #     node = YourNodeClass()
    #     self.assertTrue(hasattr(node, YourNodeClass.FUNCTION))
    #     self.assertTrue(callable(getattr(node, YourNodeClass.FUNCTION)))

    # def test_return_types(self):
    #     """RETURN_TYPES should be a non-empty tuple."""
    #     self.assertIsInstance(YourNodeClass.RETURN_TYPES, tuple)
    #     self.assertGreater(len(YourNodeClass.RETURN_TYPES), 0)

    # def test_category(self):
    #     """CATEGORY should be set."""
    #     self.assertTrue(hasattr(YourNodeClass, "CATEGORY"))
    #     self.assertIsInstance(YourNodeClass.CATEGORY, str)

    pass  # Remove this line when you uncomment tests above


if __name__ == "__main__":
    unittest.main()
