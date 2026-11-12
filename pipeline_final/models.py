"""Stronger U-Net variant: 3 downsample levels (vs shipped 2), same style."""
import torch
from torch import nn


def _block(ni, no):
    return nn.Sequential(
        nn.Conv2d(ni, no, 3, padding=1, bias=False), nn.BatchNorm2d(no), nn.SiLU(),
        nn.Conv2d(no, no, 3, padding=1, bias=False), nn.BatchNorm2d(no), nn.SiLU())


def _up(x, ref):
    return nn.functional.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)


class DeepUNet(nn.Module):
    """base=16 -> channels 16/32/64/128, ~1.9M params. Fully convolutional."""

    def __init__(self, base=16):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.e4 = _block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = _block(base * 12, base * 4)
        self.u2 = _block(base * 6, base * 2)
        self.u1 = _block(base * 3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        d = self.e4(self.pool(c))
        y = self.u3(torch.cat((_up(d, c), c), 1))
        y = self.u2(torch.cat((_up(y, b), b), 1))
        return self.head(self.u1(torch.cat((_up(y, a), a), 1)))


def param_count(net):
    return sum(p.numel() for p in net.parameters())


class SMPGrayUNet(nn.Module):
    """Segmentation-models U-Net with a (optionally ImageNet-pretrained)
    encoder. Accepts the same 1-channel 0..1 input as DeepUNet: repeats to
    3 channels and applies ImageNet normalization internally."""

    def __init__(self, encoder="resnet34", weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.encoder_name = encoder
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=weights,
                            in_channels=3, classes=1, activation=None)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.net((x - self.img_mean) / self.img_std)


def build_model_from_ckpt(ck):
    arch = str(ck.get("arch", "DeepUNet"))
    if arch.startswith("SMPUNet:"):
        return SMPGrayUNet(encoder=arch.split(":", 1)[1], weights=None)
    return DeepUNet(base=ck["base"])
