import math
from textwrap import fill

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils.data import denormalize


def _show_tensor_image(ax, image, title=None):
    image = denormalize(image.detach().float().cpu())
    if image.size(0) == 1:
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis("off")
    if title:
        ax.set_title(title)


def show_pairs(sketches, photos, n=4):
    n = min(n, sketches.size(0))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        _show_tensor_image(axes[0, i], sketches[i], "sketch")
        _show_tensor_image(axes[1, i], photos[i], "target photo")
    plt.tight_layout()
    return fig


@torch.no_grad()
def show_forward_diffusion(diffusion, photo, steps=(0, 50, 150, 300, 600, 999), noise=None):
    """Pass a fixed noise tensor to couple marginals; None keeps independent draws."""
    device = diffusion.device
    photo = photo[:1].to(device)
    noise = None if noise is None else noise[:1].to(device)
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(2.6 * (len(steps) + 1), 2.8))
    _show_tensor_image(axes[0], photo[0], "clean photo")
    for ax, step in zip(axes[1:], steps):
        t = torch.tensor([step], device=device, dtype=torch.long)
        noised, _ = diffusion.add_noise(photo, t, noise=noise)
        _show_tensor_image(ax, noised[0], f"t={step}")
    plt.tight_layout()
    return fig


def show_image_rows(rows, titles, *, figsize=None):
    """Display rows of CHW images in [-1, 1], with one title per column."""
    fig, axes = plt.subplots(
        len(rows), len(titles), squeeze=False,
        figsize=figsize or (2.6 * len(titles), 2.6 * len(rows)),
    )
    for i, (axes_row, images) in enumerate(zip(axes, rows)):
        for ax, image, title in zip(axes_row, images, titles):
            _show_tensor_image(ax, image, title if i == 0 else None)
    fig.tight_layout()
    return fig


def show_model_comparison(sketches, outputs, references, *, indices):
    """One image grid: selected pairs as rows, model outputs as columns, all in [0, 1]."""
    columns = [sketches.repeat(1, 3, 1, 1), *outputs.values(), references]
    rows = torch.stack(columns, dim=1)[indices]
    grid = make_grid(rows.flatten(0, 1).float(), nrow=len(columns), padding=4, pad_value=1)
    titles = ["Input sketch", *outputs, "Paired photograph"]

    fig, ax = plt.subplots(figsize=(3.4 * len(columns), 3 * len(indices)), layout="constrained")
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), extent=(0, len(columns), len(indices), 0))
    ax.set_xticks([i + 0.5 for i in range(len(columns))], [fill(title, 28) for title in titles], fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(indices))], [f"Pair {i + 1}" for i in indices])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_frame_on(False)
    return fig


def show_reverse_trajectory(trajectory, n=8):
    if not trajectory:
        raise ValueError("trajectory is empty")

    picks = torch.linspace(0, len(trajectory) - 1, min(n, len(trajectory))).long().tolist()
    fig, axes = plt.subplots(1, len(picks), figsize=(2.6 * len(picks), 2.8))
    if len(picks) == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        step, image = trajectory[idx]
        _show_tensor_image(ax, image[0], f"t={step}")
    plt.tight_layout()
    return fig


def show_generation_grid(sketches, generated, targets=None, n=4):
    n = min(n, sketches.size(0), generated.size(0))
    rows = []
    for i in range(n):
        sketch_rgb = sketches[i].repeat(3, 1, 1)
        rows.append(sketch_rgb)
        rows.append(generated[i])
        if targets is not None:
            rows.append(targets[i])

    per_row = 3 if targets is not None else 2
    grid = make_grid(denormalize(torch.stack(rows)), nrow=per_row, padding=2)
    fig_width = 3.2 * per_row
    fig_height = max(3.0, 2.8 * math.ceil(len(rows) / per_row))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(grid.permute(1, 2, 0).cpu())
    ax.axis("off")
    ax.set_title("sketch | generated | target" if targets is not None else "sketch | generated")
    plt.tight_layout()
    return fig


def plot_losses(history):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, linewidth=1.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig
