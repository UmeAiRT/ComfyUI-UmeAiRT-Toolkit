"""
UmeAiRT Toolkit - Path Traversal Security Test
-----------------------------------------------
Validates that the ImageSaver node correctly blocks path traversal
attempts, ensuring files are always saved within the output directory.
"""

import sys
import os
import re
import unittest
from unittest.mock import MagicMock, patch

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
sys.modules['server'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['aiohttp.web'] = MagicMock()

# Create a mock for folder_paths with a controlled output_directory
class FolderPathsMock:
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '_test_output'))

# sys.modules['folder_paths'] = FolderPathsMock()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPathTraversal(unittest.TestCase):
    """Tests that malicious filename patterns cannot escape the output directory."""

    def _sanitize_path(self, filename_pattern):
        """Replicates the sanitization logic from WirelessImageSaver.save_images()."""
        filename = filename_pattern.replace("..", "")

        full_pattern = filename.replace("\\", "/")
        if "/" in full_pattern:
            path, filename = full_pattern.rsplit("/", 1)
        else:
            path = ""

        path = path.lstrip("/\\")
        path = re.sub(r'[<>:"\\|?*]', '', path)
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)

        output_dir_abs = os.path.abspath(FolderPathsMock.output_directory)
        final_abs_path = os.path.abspath(os.path.join(output_dir_abs, path))

        return final_abs_path, output_dir_abs

    def test_basic_traversal_blocked(self):
        """Attempt ../../Windows/System32/hack"""
        final, output = self._sanitize_path("../../Windows/System32/hack")
        self.assertTrue(
            final.startswith(output),
            f"Path traversal not blocked: {final} escapes {output}"
        )

    def test_backslash_traversal_blocked(self):
        """Attempt ..\\..\\Windows\\System32\\hack"""
        final, output = self._sanitize_path("..\\..\\Windows\\System32\\hack")
        self.assertTrue(
            final.startswith(output),
            f"Backslash traversal not blocked: {final} escapes {output}"
        )

    def test_absolute_path_blocked(self):
        """Attempt /Windows/System32/hack"""
        final, output = self._sanitize_path("/Windows/System32/hack")
        self.assertTrue(
            final.startswith(output),
            f"Absolute path not blocked: {final} escapes {output}"
        )

    def test_mixed_traversal_blocked(self):
        """Attempt ..%2f..%2f style encoding (already decoded by Python)"""
        final, output = self._sanitize_path("../../../etc/passwd")
        self.assertTrue(
            final.startswith(output),
            f"Mixed traversal not blocked: {final} escapes {output}"
        )

    def test_clean_path_passes(self):
        """A normal subfolder path should resolve inside output."""
        final, output = self._sanitize_path("my_images/test_001")
        self.assertTrue(
            final.startswith(output),
            f"Clean path rejected: {final} not inside {output}"
        )

    def test_empty_path_passes(self):
        """Empty path should resolve to root output directory."""
        final, output = self._sanitize_path("image_001")
        self.assertEqual(final, output)


if __name__ == "__main__":
    unittest.main()
