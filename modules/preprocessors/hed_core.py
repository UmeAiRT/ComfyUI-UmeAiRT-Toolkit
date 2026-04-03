import torch
import numpy as np
import cv2

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


def apply_hed(image_tensor: torch.Tensor, model_path: str) -> torch.Tensor:
    import comfy.model_management as mm
    device = mm.get_torch_device()

    net = ControlNetHED_Apache2()
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    net.float().eval().to(device)

    out_batch = []
    
    for i in range(image_tensor.shape[0]):
        img_np = (image_tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.float32)
        
        H, W, C = img_np.shape
        if C == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        with torch.no_grad():
            image_hed = torch.from_numpy(img_np).float().to(device)
            # Rearrange to 1, C, H, W
            image_hed = image_hed.permute(2, 0, 1).unsqueeze(0)
            
            edges = net(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        # Convert back to 3 channels for ControlNet input
        edge_3c = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
        out_tensor = torch.from_numpy(edge_3c).float() / 255.0
        out_batch.append(out_tensor)

    del net
    mm.soft_empty_cache()

    return torch.stack(out_batch, dim=0)
