import sys
import os
import unittest

# Force UTF-8 encoding for standard output to prevent emoji print crashes in headless tests
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
from unittest.mock import MagicMock

# Add the custom_nodes folder and ComfyUI root to sys.path
comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
custom_nodes = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if comfy_root not in sys.path: sys.path.insert(0, comfy_root)
if custom_nodes not in sys.path: sys.path.insert(0, custom_nodes)

# Mock 'server' and related ComfyUI imports globally to prevent loading issues
sys.modules['server'] = MagicMock()
sys.modules['app'] = MagicMock()
sys.modules['app.frontend_management'] = MagicMock()
sys.modules['utils.install_util'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['aiohttp.web'] = MagicMock()

# Add the parent directory directly to sys.path to access the toolkit's root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSmoke(unittest.TestCase):
    def test_imports_and_mappings(self):
        """Minimal smoke test to ensure __init__.py can be imported and NODE_CLASS_MAPPINGS is populated."""
        import importlib.util
        init_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '__init__.py'))
        spec = importlib.util.spec_from_file_location("umeairt_toolkit", init_path)
        umeairt_init = importlib.util.module_from_spec(spec)
        
        # Set package name to allow relative imports
        umeairt_init.__package__ = "umeairt_toolkit"
        umeairt_init.__name__ = "umeairt_toolkit"
        sys.modules["umeairt_toolkit"] = umeairt_init
        
        # Map modules inside the toolkit so relative imports find them
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        
        # Override logger to prevent encoding/colorama crashes in headless environment
        import modules.logger
        modules.logger.log_node = lambda *args, **kwargs: None
        
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
