import torch
import torch.nn as nn
import numpy as np
import cv2

norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

def apply_lineart(image_tensor: torch.Tensor, model_path: str, coarse_model_path: str, use_coarse: bool = False) -> torch.Tensor:
    import comfy.model_management as mm
    device = mm.get_torch_device()

    model = Generator(3, 1, 3)
    
    if use_coarse:
        model.load_state_dict(torch.load(coarse_model_path, map_location='cpu', weights_only=True))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
    model.eval().to(device)

    out_batch = []
    
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.float32)
        
        H, W, C = img_np.shape
        if C == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
        # Optional: resize input to mutiples of 64 or 512, keeping original sizes for now
        # Lineart typically requires divisible dims, but ComfyUI tensors are usually fine.
        
        with torch.no_grad():
            image = torch.from_numpy(img_np).float().to(device)
            image = image / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)
            
            line = model(image)[0][0]
            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        # Invert lineart (white lines on black bg) for ControlNet
        # Wait, the original controlnet_aux does: remove_pad(255 - detected_map) which inverts it!
        edge = 255 - line
        
        edge_3c = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        out_tensor = torch.from_numpy(edge_3c).float() / 255.0
        out_batch.append(out_tensor)

    del model
    mm.soft_empty_cache()

    return torch.stack(out_batch, dim=0)
