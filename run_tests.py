import sys
from unittest.mock import MagicMock

# Force UTF-8 encoding for headless environments
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mock ComfyUI modules completely
sys.modules['comfy'] = MagicMock()
comfy_utils_mock = MagicMock()
comfy_utils_mock.model_trange = lambda *args, **kwargs: range(*args)
sys.modules['comfy.utils'] = comfy_utils_mock
sys.modules['comfy.sd'] = MagicMock()
sys.modules['comfy.sd1_clip'] = MagicMock()
sys.modules['comfy.samplers'] = MagicMock()
sys.modules['comfy.sample'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()
sys.modules['nodes'] = MagicMock()
fp_mock = MagicMock()
fp_mock.supported_pt_extensions = {".pt", ".bin", ".safetensors", ".ckpt"}
fp_mock.get_full_path = lambda f, n: f"/{f}/{n}"
fp_mock.get_filename_list = lambda f: ["model.safetensors", "model.pt"]
fp_mock.folder_names_and_paths = {"checkpoints": (["/mock/models"], set()), "embeddings": (["/mock/embeds"], set())}
fp_mock.output_directory = "/fake/out"
fp_mock.models_dir = "/fake/models"
sys.modules['folder_paths'] = fp_mock
sys.modules['psutil'] = MagicMock()
tqdm_mock = MagicMock()
tqdm_mock.tqdm = lambda *a, **kw: MagicMock()
sys.modules['tqdm'] = tqdm_mock
sys.modules['tqdm.auto'] = tqdm_mock
sys.modules['tqdm._tqdm_pandas'] = MagicMock()

# Mock heavy math dependencies that cause WMI deadlocks internally
class DummyTorch:
    def __init__(self):
        self._cache = {}
    def no_grad(self):
        def decorator(func):
            return func
        return decorator
    def __getattr__(self, name):
        if name not in self._cache:
            self._cache[name] = MagicMock()
        return self._cache[name]

sys.modules['torch'] = DummyTorch()
print("[DEBUG] run_tests.py setting sys.modules['torch'] = DummyTorch")
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()

# Lazy loads
sys.modules['comfy.k_diffusion'] = MagicMock()
sys.modules['comfy.k_diffusion.sampling'] = MagicMock()
sys.modules['comfy.k_diffusion.utils'] = MagicMock()
sys.modules['comfy.model_patcher'] = MagicMock()
sys.modules['comfy.model_sampling'] = MagicMock()
sys.modules['comfy_extras'] = MagicMock()
sys.modules['comfy_extras.nodes_upscale_model'] = MagicMock()
sys.modules['comfy_extras.nodes_custom_sampler'] = MagicMock()
sys.modules['comfy_extras.nodes_post_processing'] = MagicMock()

import unittest

if __name__ == '__main__':
    # Run unittest discovery
    unittest.main(module=None, argv=['run_tests.py', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'])
