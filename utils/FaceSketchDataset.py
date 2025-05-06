import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FaceSketchDataset(Dataset):
    """Load (sketch, photo) pairs and apply identical spatial augmentations.

    Default pipeline:
        • resize ⇒ random‑jitter crop ⇒ random flip
        • ToTensor ⇒ Normalize to [‑1,1]
    """

    def __init__(self, root_dir: str, *,
                 photo_folder: str = "photos",
                 sketch_folder: str = "sketches",
                 image_size: tuple[int, int] = (256, 256),
                 transform: transforms.Compose | None = None):
        self.photo_dir = os.path.join(root_dir, photo_folder)
        self.sketch_dir = os.path.join(root_dir, sketch_folder)

        photo_ext = (".jpg", ".jpeg", ".png")
        self.photo_files = [f for f in os.listdir(self.photo_dir) if f.lower().endswith(photo_ext)]
        self.sketch_files = [f for f in os.listdir(self.sketch_dir) if f.lower().endswith(photo_ext)]

        # Pair by basename
        self.pairs: list[tuple[str, str]] = []
        for photo in self.photo_files:
            base = os.path.splitext(photo)[0]
            sketch_name = f"{base}-sz1.jpg"  # adjust if naming differs
            if sketch_name in self.sketch_files:
                self.pairs.append((photo, sketch_name))

        if transform is None:
            common_tf = [
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
            ]
            self.photo_transform = transforms.Compose(
                common_tf + [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            self.sketch_transform = transforms.Compose(
                common_tf + [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            # user supplied single transform (must output Tensor already normalised)
            self.photo_transform = self.sketch_transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        photo_fn, sketch_fn = self.pairs[idx]
        photo = Image.open(os.path.join(self.photo_dir, photo_fn)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_fn)).convert("L")

        photo = self.photo_transform(photo)
        sketch = self.sketch_transform(sketch)
        return sketch, photo  # shape (1, H, W), (3, H, W)
