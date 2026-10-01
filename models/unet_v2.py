"""A deeper conditional diffusion U-Net; models.unet keeps the original baseline.

Scale-shift normalization and residual resampling follow the ADM design:
https://arxiv.org/abs/2105.05233
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import SketchEncoder, SelfAttention2d, get_timestep_embedding, make_norm


class AdaptiveResBlock(nn.Module):
    """Time-conditioned normalization: (1 + scale(t)) * GroupNorm(h) + shift(t)."""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, resample=None):
        super().__init__()
        self.norm1 = make_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, 2 * out_channels)
        self.norm2 = make_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.resample = resample
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(x))
        if self.resample == "down":
            h, x = F.avg_pool2d(h, 2), F.avg_pool2d(x, 2)
        elif self.resample == "up":
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        scale, shift = self.time_proj(F.silu(time_emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.conv2(self.dropout(F.silu(h)))


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.down = AdaptiveResBlock(in_channels, out_channels, time_emb_dim, dropout, resample="down")
        self.refine = AdaptiveResBlock(out_channels, out_channels, time_emb_dim, dropout)

    def forward(self, x, sketch_features, time_emb):
        x = self.down(x, time_emb) + sketch_features
        return self.refine(x, time_emb)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.up = AdaptiveResBlock(in_channels, out_channels, time_emb_dim, dropout, resample="up")
        self.fuse = AdaptiveResBlock(3 * out_channels, out_channels, time_emb_dim, dropout)
        self.refine = AdaptiveResBlock(out_channels, out_channels, time_emb_dim, dropout)

    def forward(self, x, skip, sketch_features, time_emb):
        x = self.up(x, time_emb)
        x = self.fuse(torch.cat([x, skip, sketch_features], dim=1), time_emb)
        return self.refine(x, time_emb)


class UNetV2(nn.Module):
    """Two residual blocks per scale, direct decoder conditioning, and low-resolution attention.

    Inputs are [sketch, noisy RGB photo], with spatial dimensions divisible by 16.
    """

    def __init__(
        self,
        n_channels=4,
        n_classes=3,
        *,
        base_channels=48,
        time_emb_dim=192,
        sketch_channels=1,
        dropout=0.1,
        use_attention=True,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.sketch_channels = sketch_channels
        # Store only constructor arguments so the notebooks can reload this architecture.
        self.model_args = dict(n_channels=n_channels, n_classes=n_classes,
                               base_channels=base_channels, time_emb_dim=time_emb_dim,
                               sketch_channels=sketch_channels, dropout=dropout,
                               use_attention=use_attention)
        channels = [base_channels * m for m in (1, 2, 4, 8, 8)]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 4 * time_emb_dim), nn.SiLU(),
            nn.Linear(4 * time_emb_dim, time_emb_dim),
        )
        self.sketch_encoder = SketchEncoder(sketch_channels, tuple(channels))
        self.input_block = AdaptiveResBlock(n_channels, channels[0], time_emb_dim, dropout)
        self.input_refine = AdaptiveResBlock(channels[0], channels[0], time_emb_dim, dropout)
        self.encoder = nn.ModuleList([
            EncoderStage(channels[i], channels[i + 1], time_emb_dim, dropout)
            for i in range(4)
        ])
        self.encoder_attention = SelfAttention2d(channels[3]) if use_attention else nn.Identity()
        self.middle_in = AdaptiveResBlock(channels[4], channels[4], time_emb_dim, dropout)
        self.middle_attention = SelfAttention2d(channels[4]) if use_attention else nn.Identity()
        self.middle_out = AdaptiveResBlock(channels[4], channels[4], time_emb_dim, dropout)
        self.decoder = nn.ModuleList([
            DecoderStage(channels[i + 1], channels[i], time_emb_dim, dropout)
            for i in reversed(range(4))
        ])
        self.decoder_attention = SelfAttention2d(channels[3]) if use_attention else nn.Identity()
        self.output = nn.Sequential(make_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], n_classes, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        for attention in (self.encoder_attention, self.middle_attention, self.decoder_attention):
            if isinstance(attention, SelfAttention2d):
                nn.init.zeros_(attention.attn.out_proj.weight)
                nn.init.zeros_(attention.attn.out_proj.bias)

    def forward(self, x, t):
        """x contains [s, x_t] along channels; return v_theta(x_t, s, t), not an image density."""
        time_emb = self.time_mlp(get_timestep_embedding(t, self.time_emb_dim))
        sketch_features = self.sketch_encoder(x[:, :self.sketch_channels])
        x = self.input_block(x, time_emb) + sketch_features[0]
        x = self.input_refine(x, time_emb)
        skips = [x]
        for level, stage in enumerate(self.encoder, start=1):
            x = stage(x, sketch_features[level], time_emb)
            if level == 3:
                x = self.encoder_attention(x)
            skips.append(x)
        x = self.middle_in(x, time_emb)
        x = self.middle_attention(x)
        x = self.middle_out(x, time_emb)
        for stage, level in zip(self.decoder, reversed(range(4))):
            x = stage(x, skips[level], sketch_features[level], time_emb)
            if level == 3:
                x = self.decoder_attention(x)
        return self.output(x)
