import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return sinusoidal embeddings of shape (B, dim)."""
    half = dim // 2
    freq_const = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=timesteps.device) * -freq_const)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def make_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    for groups in (max_groups, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResBlock(nn.Module):
    """Small diffusion residual block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = make_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = make_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SketchBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            make_norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            make_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SketchEncoder(nn.Module):
    """Deterministic sketch features at each scale, not a variational q(z | x)."""

    def __init__(self, sketch_channels: int, channels: tuple[int, int, int, int, int]):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.s1 = SketchBlock(sketch_channels, c1)
        self.s2 = SketchBlock(c1, c2, downsample=True)
        self.s3 = SketchBlock(c2, c3, downsample=True)
        self.s4 = SketchBlock(c3, c4, downsample=True)
        self.s5 = SketchBlock(c4, c5, downsample=True)

    def forward(self, sketch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s1 = self.s1(sketch)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        return s1, s2, s3, s4, s5


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ResBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x), time_emb)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_emb_dim: int,
        *,
        bilinear: bool = True,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.block = ResBlock(in_channels + skip_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.block(torch.cat([skip, x], dim=1), time_emb)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = make_norm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        tokens, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + tokens.transpose(1, 2).reshape(b, c, h, w)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        *,
        bilinear: bool = True,
        time_emb_dim: int = 128,
        base_channels: int = 64,
        sketch_channels: int = 1,
        use_attention: bool = True,
    ):
        """Diffusion U-Net for sketch-conditioned photo generation.

        The first `sketch_channels` channels are treated as the sketch condition.
        With the default notebook setup this means input is [sketch, noisy_photo].
        """
        super().__init__()
        self.sketch_channels = sketch_channels
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 8 if bilinear else base_channels * 16

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.time_emb_dim = time_emb_dim

        self.sketch_encoder = SketchEncoder(sketch_channels, (c1, c2, c3, c4, c5))
        self.inc = ResBlock(n_channels, c1, time_emb_dim)
        self.down1 = Down(c1, c2, time_emb_dim)
        self.down2 = Down(c2, c3, time_emb_dim)
        self.down3 = Down(c3, c4, time_emb_dim)
        self.down4 = Down(c4, c5, time_emb_dim)
        self.attn = SelfAttention2d(c5) if use_attention else nn.Identity()
        self.up1 = Up(c5, c4, c4, time_emb_dim, bilinear=bilinear)
        self.up2 = Up(c4, c3, c3, time_emb_dim, bilinear=bilinear)
        self.up3 = Up(c3, c2, c2, time_emb_dim, bilinear=bilinear)
        self.up4 = Up(c2, c1, c1, time_emb_dim, bilinear=bilinear)
        self.outc = OutConv(c1, n_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x contains [s, x_t] along channels; return v_theta(x_t, s, t), an RGB-shaped tensor."""
        if x.size(1) < self.sketch_channels:
            raise ValueError(f"Expected at least {self.sketch_channels} input channel(s), got {x.size(1)}")

        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_mlp(time_emb)
        sketch_feats = self.sketch_encoder(x[:, : self.sketch_channels])

        x1 = self.inc(x, time_emb) + sketch_feats[0]
        x2 = self.down1(x1, time_emb) + sketch_feats[1]
        x3 = self.down2(x2, time_emb) + sketch_feats[2]
        x4 = self.down3(x3, time_emb) + sketch_feats[3]
        x5 = self.down4(x4, time_emb) + sketch_feats[4]
        x5 = self.attn(x5)

        x = self.up1(x5, x4, time_emb)
        x = self.up2(x, x3, time_emb)
        x = self.up3(x, x2, time_emb)
        x = self.up4(x, x1, time_emb)
        return self.outc(x)
