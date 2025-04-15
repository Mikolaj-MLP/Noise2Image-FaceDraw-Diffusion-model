import torch
from torch.utils.data import Dataset

def no_op_transform(sketch, photo):
    """Return the sketch/photo pair unchanged."""
    return sketch, photo

def horizontal_flip(sketch, photo):
    """
    Flip each image horizontally. shape (C, H, W),
    so flipping across the width dimension => dim=2.
    """
    sketch_flipped = torch.flip(sketch, dims=[2])  
    photo_flipped = torch.flip(photo, dims=[2])
    return sketch_flipped, photo_flipped

def rotate_90(sketch, photo):
    """
    Rotate each image 90 degrees (k=1).
    rotate along dims [1, 2].
    """
    sketch_rot = torch.rot90(sketch, k=1, dims=[1, 2])
    photo_rot = torch.rot90(photo, k=1, dims=[1, 2])
    return sketch_rot, photo_rot

def rotate_180(sketch, photo):
    """
    Rotate each image 180 degrees (k=2).
    """
    sketch_rot = torch.rot90(sketch, k=2, dims=[1, 2])
    photo_rot = torch.rot90(photo, k=2, dims=[1, 2])
    return sketch_rot, photo_rot

class ExpandDataset(Dataset):
    """
    Wraps an existing dataset (e.g., FaceSketchDataset) and expands it by applying
    a fixed list of deterministic transforms to each sample, effectively multiplying
    dataset length by len(transform_fns).
    """
    def __init__(self, base_dataset, transform_fns):
        """
        :param base_dataset: an instance of FaceSketchDataset (or similar),
                             returning (sketch_tensor, photo_tensor).
        :param transform_fns: a list of callables, each taking (sketch, photo)
                              and returning (sketch_aug, photo_aug).
        """
        self.base_dataset = base_dataset
        self.transform_fns = transform_fns
        self.num_transforms = len(transform_fns)

    def __len__(self):
        return len(self.base_dataset) * self.num_transforms

    def __getitem__(self, idx):
        base_idx = idx // self.num_transforms
        tf_idx = idx % self.num_transforms

        sketch, photo = self.base_dataset[base_idx]  

        # Apply the chosen transform
        transform_fn = self.transform_fns[tf_idx]
        sketch_aug, photo_aug = transform_fn(sketch, photo)

        return sketch_aug, photo_aug
