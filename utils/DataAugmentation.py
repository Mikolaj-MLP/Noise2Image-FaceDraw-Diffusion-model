import torch
from torch.utils.data import Dataset

class ExpandDataset(Dataset):
    """
    Wraps an existing dataset (e.g., FaceSketchDataset) and expands it by applying
    a fixed list of deterministic transforms to each sample, effectively multiplying
    dataset length by len(transform_fns).

    No images are saved to disk. Everything happens in memory on the fly.
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
        # e.g. if base dataset has 100 samples, and we have 4 transforms,
        # total = 100 * 4 = 400
        return len(self.base_dataset) * self.num_transforms

    def __getitem__(self, idx):
        # figure out which base sample and which transform
        base_idx = idx // self.num_transforms
        tf_idx = idx % self.num_transforms

        # retrieve (sketch, photo) from the base dataset
        sketch, photo = self.base_dataset[base_idx]  # both are Tensors

        # apply the chosen transform
        transform_fn = self.transform_fns[tf_idx]
        sketch_aug, photo_aug = transform_fn(sketch, photo)

        # return the augmented pair
        return sketch_aug, photo_aug
