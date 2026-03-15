import sys
import os
import unittest
import importlib.util
from unittest.mock import MagicMock, patch

# Force UTF-8 encoding for standard output to prevent emoji print crashes in headless tests
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add the custom_nodes folder and ComfyUI root to sys.path
comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
custom_nodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if comfy_root not in sys.path: sys.path.insert(0, comfy_root)
if custom_nodes not in sys.path: sys.path.insert(0, custom_nodes)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimpleFolderPaths:
    def __init__(self):
        self.folder_names_and_paths = {}
        self.models_dir = "mock_models_dir"
        self.supported_pt_extensions = {".pt", ".bin", ".safetensors"}
    def get_filename_list(self, folder):
        return []
    def add_model_folder_path(self, folder, path):
        pass


class TestSmoke(unittest.TestCase):
    def test_imports_and_mappings(self):
        """Minimal smoke test to ensure __init__.py can be imported and NODE_CLASS_MAPPINGS is populated."""
        
        # Override logger to prevent encoding/colorama crashes in headless environment
        import modules.logger
        modules.logger.log_node = lambda *args, **kwargs: None

        # Build comprehensive mapping for ComfyUI's internal modules
        mocks = {
            'server': MagicMock(),
            'app': MagicMock(),
            'app.frontend_management': MagicMock(),
            'utils.install_util': MagicMock(),
            'aiohttp': MagicMock(),
            'aiohttp.web': MagicMock(),
            'comfy': MagicMock(),
            'comfy.sd1_clip': MagicMock(),
            'comfy.model_management': MagicMock(),
            'nodes': MagicMock(),
            'folder_paths': SimpleFolderPaths()
        }

        with patch.dict('sys.modules', mocks):
            init_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '__init__.py'))
            spec = importlib.util.spec_from_file_location("umeairt_toolkit", init_path)
            umeairt_init = importlib.util.module_from_spec(spec)
            
            # Set package name to allow relative imports
            umeairt_init.__package__ = "umeairt_toolkit"
            umeairt_init.__name__ = "umeairt_toolkit"
            sys.modules["umeairt_toolkit"] = umeairt_init
            
            try:
                spec.loader.exec_module(umeairt_init)
            except Exception as e:
                import traceback
                with open("error_dump.txt", "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
                self.fail(f"Smoke test failed during import: {e}")
                
            self.assertTrue(hasattr(umeairt_init, 'NODE_CLASS_MAPPINGS'), "NODE_CLASS_MAPPINGS must be defined")
            self.assertTrue(hasattr(umeairt_init, 'NODE_DISPLAY_NAME_MAPPINGS'), "NODE_DISPLAY_NAME_MAPPINGS must be defined")
            
            # Ensure mappings are not empty
            self.assertGreater(len(umeairt_init.NODE_CLASS_MAPPINGS), 0, "NODE_CLASS_MAPPINGS should contain registered nodes")
            
            print(f"Smoke test passed: {len(umeairt_init.NODE_CLASS_MAPPINGS)} nodes mapped successfully.")

if __name__ == "__main__":
    unittest.main()
