import os
import base64

# A small 1x1 transparent PNG as a placeholder
b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

assets_dir = r"c:\AI\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-UmeAiRT-Toolkit\assets"
os.makedirs(assets_dir, exist_ok=True)
path = os.path.join(assets_dir, "signature.png")

if not os.path.exists(path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Created placeholder at {path}")
else:
    print("signature.png already exists.")
