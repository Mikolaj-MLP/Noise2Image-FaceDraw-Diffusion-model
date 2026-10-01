"""Pixel-space diffusion: x_0 is a photo, s a sketch, and x_t a noisy photo.

q denotes the fixed forward noising process; p_theta denotes the learned
reverse process. We model p_theta(x_0 | s), the distribution of photos given
a sketch, rather than evaluate a density p(x) at an input image.
Noise-level index 0 is the first noised state (t=1 in the usual equations).
"""
import math

import torch

def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.arange(timesteps + 1, dtype=torch.float32)
    alphas_cum = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1 - alphas_cum[1:] / alphas_cum[:-1]
    return betas.clamp(max=0.999)


def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 2e-2, timesteps)


class Diffusion:
    def __init__(self, timesteps=1000, device="cpu", schedule="cosine"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule!r}")

        self.device = torch.device(device)
        self.timesteps = timesteps
        self.betas = betas.to(self.device)
        self.alphas = 1 - self.betas
        # alpha_hat[i] = alpha_bar_{i+1} = product_{j=0}^i (1 - betas[j]).
        self.alpha_hat = torch.cumprod(self.alphas, 0)

        alpha_hat_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_hat[:-1]])
        self.sqrt_alpha_hat = self.alpha_hat.sqrt()
        self.sqrt_one_minus_alpha_hat = (1 - self.alpha_hat).sqrt()
        self.sqrt_recip_alphas = (1 / self.alphas).sqrt()
        # beta_tilde_t: variance of q(x_{t-1} | x_t, x_0), also used for reverse sampling.
        self.posterior_variance = self.betas * (1 - alpha_hat_prev) / (1 - self.alpha_hat)

    def to(self, device):
        self.device = torch.device(device)
        for name in (
            "betas",
            "alphas",
            "alpha_hat",
            "sqrt_alpha_hat",
            "sqrt_one_minus_alpha_hat",
            "sqrt_recip_alphas",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(self.device))
        return self

    @staticmethod
    def _extract(values, t, x_shape):
        """Select one coefficient per image; reshape to (B, 1, 1, 1) for broadcasting."""
        out = values.gather(0, t.to(values.device).long())
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x, t, noise=None):
        """Draw x_t from q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I).

        Here x is the clean x_0; noise is epsilon ~ N(0, I).
        """
        noise = torch.randn_like(x) if noise is None else noise
        sqrt_alpha = self._extract(self.sqrt_alpha_hat, t, x.shape)
        sqrt_sigma = self._extract(self.sqrt_one_minus_alpha_hat, t, x.shape)
        return sqrt_alpha * x + sqrt_sigma * noise, noise

    q_sample = add_noise

    def v_target(self, x0, noise, t):
        """v = sqrt(alpha_bar_t)*epsilon - sqrt(1-alpha_bar_t)*x_0."""
        sqrt_alpha = self._extract(self.sqrt_alpha_hat, t, x0.shape)
        sqrt_sigma = self._extract(self.sqrt_one_minus_alpha_hat, t, x0.shape)
        return sqrt_alpha * noise - sqrt_sigma * x0

    def training_target(self, x0, noise, t, prediction_type="v"):
        if prediction_type in {"v", "velocity"}:
            return self.v_target(x0, noise, t)
        if prediction_type in {"eps", "epsilon", "noise"}:
            return noise
        raise ValueError(f"Unknown prediction type: {prediction_type!r}")

    def predict_x0_from_noise(self, x_t, t, noise):
        sqrt_alpha = self._extract(self.sqrt_alpha_hat, t, x_t.shape)
        sqrt_sigma = self._extract(self.sqrt_one_minus_alpha_hat, t, x_t.shape)
        return (x_t - sqrt_sigma * noise) / sqrt_alpha

    def predict_x0_from_v(self, x_t, t, v):
        """Invert the v parameterization: x_0_hat = sqrt(alpha_bar_t)*x_t - sqrt(1-alpha_bar_t)*v_hat."""
        sqrt_alpha = self._extract(self.sqrt_alpha_hat, t, x_t.shape)
        sqrt_sigma = self._extract(self.sqrt_one_minus_alpha_hat, t, x_t.shape)
        return sqrt_alpha * x_t - sqrt_sigma * v

    def predict_noise_from_v(self, x_t, t, v):
        sqrt_alpha = self._extract(self.sqrt_alpha_hat, t, x_t.shape)
        sqrt_sigma = self._extract(self.sqrt_one_minus_alpha_hat, t, x_t.shape)
        return sqrt_sigma * x_t + sqrt_alpha * v

    def model_predictions(self, x_t, t, model_output, prediction_type="v"):
        if prediction_type in {"v", "velocity"}:
            pred_noise = self.predict_noise_from_v(x_t, t, model_output)
            pred_x0 = self.predict_x0_from_v(x_t, t, model_output)
        elif prediction_type in {"eps", "epsilon", "noise"}:
            pred_noise = model_output
            pred_x0 = self.predict_x0_from_noise(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type!r}")
        return pred_noise, pred_x0.clamp(-1, 1)

    @torch.no_grad()
    def p_sample(self, model, sketch, x_t, t, prediction_type="v"):
        """One DDPM step: p_theta(x_{t-1} | x_t, s) = N(mu_theta, beta_tilde_t*I).

        The predicted noise determines mu_theta; the variance is fixed.
        """
        model_output = model(torch.cat([sketch, x_t], dim=1), t)
        pred_noise, pred_x0 = self.model_predictions(x_t, t, model_output, prediction_type)

        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_hat_t = self._extract(self.sqrt_one_minus_alpha_hat, t, x_t.shape)
        mean = sqrt_recip_alpha_t * (x_t - beta_t * pred_noise / sqrt_one_minus_alpha_hat_t)

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(t.shape[0], *((1,) * (x_t.ndim - 1)))
        variance = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + nonzero_mask * variance.sqrt() * noise, pred_x0

    @torch.no_grad()
    def sample(self, model, sketch, *, prediction_type="v", channels=3):
        """Start from p(x_T)=N(0,I) and sample each reverse transition.

        Their marginal defines p_theta(x_0 | s):
        integral p(x_T) * product_{t=1}^T p_theta(x_{t-1} | x_t, s) dx_{1:T}.
        """
        model.eval()
        sketch = sketch.to(self.device)
        x = torch.randn(
            sketch.size(0), channels, sketch.size(2), sketch.size(3), device=self.device
        )
        for step in reversed(range(self.timesteps)):
            t = torch.full((sketch.size(0),), step, device=self.device, dtype=torch.long)
            x, _ = self.p_sample(model, sketch, x, t, prediction_type=prediction_type)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        sketch,
        *,
        steps=50,
        eta=0.0,
        prediction_type="v",
        channels=3,
        noise=None,
        return_intermediates=False,
    ):
        """DDIM follows the same denoiser over fewer noise levels.

        With eta=0, the initial noise determines the result; no new noise is added.
        """
        if steps < 1:
            raise ValueError("DDIM needs at least one sampling step.")

        model.eval()
        sketch = sketch.to(self.device)
        if noise is None:
            x = torch.randn(
                sketch.size(0), channels, sketch.size(2), sketch.size(3), device=self.device
            )
        else:
            x = noise.to(self.device)
        steps = min(steps, self.timesteps)
        sample_steps = torch.linspace(self.timesteps - 1, 0, steps, device=self.device).long()
        intermediates = []

        for i, step in enumerate(sample_steps):
            t = torch.full((sketch.size(0),), step.item(), device=self.device, dtype=torch.long)
            model_output = model(torch.cat([sketch, x], dim=1), t)
            pred_noise, pred_x0 = self.model_predictions(x, t, model_output, prediction_type)

            if i == len(sample_steps) - 1:
                x = pred_x0
                if return_intermediates:
                    intermediates.append((step.item(), x.clamp(-1, 1).detach().cpu()))
                break

            next_step = sample_steps[i + 1]
            alpha = self.alpha_hat[step].reshape(1, 1, 1, 1)
            alpha_next = self.alpha_hat[next_step].reshape(1, 1, 1, 1)
            sigma = eta * torch.sqrt(
                ((1 - alpha_next) / (1 - alpha)) * (1 - alpha / alpha_next)
            )
            noise_scale = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0))
            noise = torch.randn_like(x) if eta > 0 else 0
            # x_next = sqrt(alpha_bar_next)*x_0_hat + sqrt(1-alpha_bar_next-sigma^2)*epsilon_hat + sigma*epsilon.
            x = alpha_next.sqrt() * pred_x0 + noise_scale * pred_noise + sigma * noise
            if return_intermediates:
                intermediates.append((next_step.item(), x.clamp(-1, 1).detach().cpu()))

        x = x.clamp(-1, 1)
        if return_intermediates:
            return x, intermediates
        return x

    def sample_timesteps(self, bsz):
        return torch.randint(0, self.timesteps, (bsz,), device=self.device)
