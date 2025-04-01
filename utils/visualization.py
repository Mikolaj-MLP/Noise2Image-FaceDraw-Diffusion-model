import matplotlib.pyplot as plt

def show_sketch_photo_batch(sketch_batch, photo_batch, num_samples=6):
    """
    Display a grid of sketch-photo pairs.
    sketch_batch: tensor (B, 1, H, W)
    photo_batch: tensor (B, 3, H, W)
    """
    sketch_batch = sketch_batch[:num_samples]
    photo_batch = photo_batch[:num_samples]
    fig, axs = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
    for i in range(num_samples):
        sketch = sketch_batch[i].squeeze().cpu().numpy()
        photo = photo_batch[i].permute(1, 2, 0).cpu().numpy()
        axs[i, 0].imshow(sketch, cmap='gray')
        axs[i, 0].set_title("Sketch")
        axs[i, 1].imshow(photo)
        axs[i, 1].set_title("Photo")
        for ax in axs[i]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
