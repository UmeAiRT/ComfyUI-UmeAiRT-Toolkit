import sys
import os
import torch
from unittest.mock import MagicMock

# Mock ComfyUI environment
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['nodes'] = MagicMock()

class FolderPathsMock:
    output_directory = os.path.abspath("C:/output_mock_dir")

sys.modules['folder_paths'] = FolderPathsMock()

# Import from toolkit
sys.path.insert(0, os.path.abspath('.'))
from modules.image_nodes import UmeAiRT_WirelessImageSaver
from modules.common import UME_SHARED_STATE

# Setup
os.makedirs("C:/output_mock_dir", exist_ok=True)
UME_SHARED_STATE["imagesize"] = {"width": 512, "height": 512}
UME_SHARED_STATE["model_name"] = "test_model"

saver = UmeAiRT_WirelessImageSaver()
img = torch.zeros((1, 512, 512, 3))

print("Testing malicious path traversal...")
# Attempt path traversal
res = saver.save_images([img], "/Windows/System32/hack")
print(f"Returned UI dict: {res}")

filenames = [i['filename'] for i in res['ui']['images']]
subfolders = [i['subfolder'] for i in res['ui']['images']]

print(f"Files saved: {filenames}")
print(f"Subfolders: {subfolders}")

if len(subfolders) > 0 and not subfolders[0].startswith("Windows"):
    print("SUCCESS: Path traversal was blocked or sanitized properly.")
else:
    print("WARNING: Path was literally evaluated as starting with Windows.")

print("Test complete.")
