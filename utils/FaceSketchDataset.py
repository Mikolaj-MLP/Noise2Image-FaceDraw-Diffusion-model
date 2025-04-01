import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FaceSketchDataset(Dataset):
    def __init__(self, root_dir, photo_folder='photos', sketch_folder='sketches',
                 image_size=(256, 256), transform=None):
        self.photo_dir = os.path.join(root_dir, photo_folder)
        self.sketch_dir = os.path.join(root_dir, sketch_folder)

        self.photo_files = [f for f in os.listdir(self.photo_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        self.sketch_files = [f for f in os.listdir(self.sketch_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

        # Build pairs based on file name patterns
        self.pairs = []
        for photo in self.photo_files:
            base_name = os.path.splitext(photo)[0]
            sketch_name = f"{base_name}-sz1.jpg"
            if sketch_name in self.sketch_files:
                self.pairs.append((photo, sketch_name))

        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        photo_filename, sketch_filename = self.pairs[idx]

        photo_path = os.path.join(self.photo_dir, photo_filename)
        sketch_path = os.path.join(self.sketch_dir, sketch_filename)

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("L")

        photo = self.transform(photo)
        sketch = self.transform(sketch)

        return sketch, photo  # (input, target)
