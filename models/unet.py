import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Positional / sinusoidal timestep embedding
# --------------------------------------------------

def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings of shape (B, dim) given integer timesteps."""
    half = dim // 2
    freq_const = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=timesteps.device) * -freq_const)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:  # zero‑pad if odd dim
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)

# --------------------------------------------------
# Building blocks
# --------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------
# U‑Net with sinusoidal time conditioning for diffusion
# --------------------------------------------------

class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, *, bilinear: bool = True, time_emb_dim: int = 128):
        """Diffusion‑conditioned U‑Net.

        Args:
            n_channels: 1 + 3 = 4 (sketch + noisy photo)
            n_classes: output channels (predicting *v* or noise) – here 3
        """
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear=bilinear)
        self.up2 = Up(512, 256 // factor, bilinear=bilinear)
        self.up3 = Up(256, 128 // factor, bilinear=bilinear)
        self.up4 = Up(128, 64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

        # Sinusoidal time embedding → small MLP → 64‑channel conv inject
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.time_conv = nn.Conv2d(time_emb_dim, 64, kernel_size=1)
        self.time_emb_dim = time_emb_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # sinusoidal embedding → MLP → broadcast to H×W
        emb = get_timestep_embedding(t, self.time_emb_dim)   # (B, dim)
        emb = self.time_mlp(emb)                             # (B, dim)
        emb = emb[:, :, None, None]                          # (B, dim, 1, 1)
        emb = self.time_conv(emb)                            # (B, 64, 1, 1)
        emb = emb.expand(-1, -1, x.size(2), x.size(3))       # (B, 64, H, W)

        x1 = self.inc(x) + emb
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
