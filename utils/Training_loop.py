import torch
import torch.nn.functional as F
from tqdm import tqdm  

class TrainingLoop:
    def __init__(self, model, diffusion, dataloader, optimizer, device, epochs=50):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for sketch, photo in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
                sketch = sketch.to(self.device)  # shape: (B, 1, H, W) 
                photo = photo.to(self.device)    # shape: (B, 3, H, W)
                batch_size = sketch.shape[0]

                t = self.diffusion.sample_timesteps(batch_size)
                noisy_photo, noise = self.diffusion.add_noise(photo, t)
                # conditional input: concatenation of sketch and noisy photo (B, 4, H, W)
                cond_input = torch.cat([sketch, noisy_photo], dim=1)
                # Predict the noise using the model
                predicted_noise = self.model(cond_input, t)
                # Compute the MSE loss between predicted and true noise
                loss = F.mse_loss(predicted_noise, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
