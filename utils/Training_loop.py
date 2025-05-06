import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm


class TrainingLoop:
    """Training wrapper with v‑prediction target and EMA weights."""

    def __init__(self, model, diffusion, dataloader, optimizer, device,
                 *, epochs: int = 50, ema_decay: float = 0.999):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.ema_decay = ema_decay
        # Exponential moving‑average of weights
        self.ema = copy.deepcopy(model).to(device).eval()

    def _update_ema(self):
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def train(self):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for sketch, photo in tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.epochs}", leave=False):
                sketch, photo = sketch.to(self.device), photo.to(self.device)
                b = sketch.size(0)
                t = self.diffusion.sample_timesteps(b)

                noisy_photo, noise = self.diffusion.add_noise(photo, t)
                sigma = torch.sqrt(1 - self.diffusion.alpha_hat[t])[:, None, None, None]
                target_v = noise / sigma  # v‑prediction target

                cond_input = torch.cat([sketch, noisy_photo], dim=1)
                v_pred = self.model(cond_input, t)
                loss = F.mse_loss(v_pred, target_v)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_ema()

                epoch_loss += loss.item()

            print(f"Epoch {epoch:>3}/{self.epochs}:  loss = {epoch_loss / len(self.dataloader):.5f}")

    # Convenience getter for sampling
    @property
    def ema_model(self):
        return self.ema
