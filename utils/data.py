import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PairedRandomTransform:
    """Shared crop, affine transform, and flip for the sketch and target face."""

    def __init__(
        self,
        image_size=256,
        resize_margin=32,
        hflip_p=0.5,
        degrees=12,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        affine=True,
    ):
        self.image_size = image_size
        self.resize_size = image_size + resize_margin
        self.hflip_p = hflip_p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.affine = affine

    def __call__(self, sketch, photo):
        size = [self.resize_size, self.resize_size]
        sketch = TF.resize(sketch, size, interpolation=InterpolationMode.BICUBIC)
        photo = TF.resize(photo, size, interpolation=InterpolationMode.BICUBIC)

        if self.affine:
            affine_params = T.RandomAffine.get_params(
                degrees=[-self.degrees, self.degrees],
                translate=self.translate,
                scale_ranges=self.scale,
                shears=None,
                img_size=[self.image_size, self.image_size],
            )
            # Keep real context until the final crop; extend edges without mirroring features.
            sketch, photo = (
                TF.center_crop(
                    TF.affine(
                        TF.pad(image, self.image_size // 2, padding_mode="edge"),
                        *affine_params,
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                    size,
                )
                for image in (sketch, photo)
            )

        i, j, h, w = T.RandomCrop.get_params(
            photo, output_size=(self.image_size, self.image_size)
        )
        sketch = TF.crop(sketch, i, j, h, w)
        photo = TF.crop(photo, i, j, h, w)

        if torch.rand(()) < self.hflip_p:
            sketch = TF.hflip(sketch)
            photo = TF.hflip(photo)

        return sketch, photo


class CenterTransform:
    """Deterministic transform for validation and visualization."""

    def __init__(self, image_size=256):
        self.image_size = image_size

    def __call__(self, sketch, photo):
        size = [self.image_size, self.image_size]
        sketch = TF.resize(sketch, size, interpolation=InterpolationMode.BICUBIC)
        photo = TF.resize(photo, size, interpolation=InterpolationMode.BICUBIC)
        return sketch, photo


class PairedManifestDataset(Dataset):
    """
    Dataset backed by a CSV manifest with:
    dataset, split, sketch_path, photo_path
    """

    def __init__(
        self,
        manifest_path,
        *,
        split="train",
        image_size=256,
        repeat=1,
        datasets=None,
        max_items=None,
        random_transform=True,
        affine=True,
    ):
        self.manifest_path = Path(manifest_path).expanduser()
        wanted_splits = {split} if isinstance(split, str) else set(split)
        wanted_datasets = None if datasets is None else set(datasets)

        rows = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] not in wanted_splits:
                    continue
                if wanted_datasets is not None and row["dataset"] not in wanted_datasets:
                    continue
                rows.append(row)

        if max_items is not None:
            rows = rows[:max_items]
        self.rows = rows
        self.repeat = repeat
        self.geo = (
            PairedRandomTransform(image_size=image_size, affine=affine)
            if random_transform
            else CenterTransform(image_size=image_size)
        )
        self.sketch_to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])
        self.photo_to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @property
    def num_pairs(self):
        return len(self.rows)

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, idx):
        row = self.rows[idx % len(self.rows)]
        sketch = Image.open(row["sketch_path"]).convert("L")
        photo = Image.open(row["photo_path"]).convert("RGB")
        sketch, photo = self.geo(sketch, photo)
        return self.sketch_to_tensor(sketch), self.photo_to_tensor(photo)


def make_dataloader(
    manifest_path,
    *,
    split="train",
    image_size=256,
    batch_size=8,
    repeat=1,
    shuffle=True,
    num_workers=2,
    datasets=None,
    max_items=None,
    random_transform=True,
    affine=True,
):
    dataset = PairedManifestDataset(
        manifest_path,
        split=split,
        image_size=image_size,
        repeat=repeat,
        datasets=datasets,
        max_items=max_items,
        random_transform=random_transform,
        affine=affine,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
    )
    return loader


def manifest_counts(manifest_path):
    counts = Counter()
    with Path(manifest_path).expanduser().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["dataset"], row["split"])] += 1
    return counts


def denormalize(x):
    return (x.clamp(-1, 1) + 1) / 2
