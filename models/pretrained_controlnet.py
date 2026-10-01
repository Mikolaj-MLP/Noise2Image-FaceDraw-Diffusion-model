"""SD 1.5 latent diffusion conditioned on sketch s and text embedding c.

q_phi(z | x) is the frozen VAE encoder distribution for a photo x.
The scaled latent z_0 = k*z, not the RGB image, is diffused here.
Generation starts from p(z_T)=N(0,I), denoises to z_0, then decodes with D_psi(z_0/k).
This defines p_theta(x | s, c) through sampling, without evaluating its density.
Only ControlNet's theta is updated here; the VAE and SD U-Net stay frozen.
"""

import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline


def freeze_backbone(pipe):
    """Keep gradients through the U-Net, but update only ControlNet parameters."""
    for module in (pipe.vae, pipe.text_encoder, pipe.unet):
        module.requires_grad_(False).eval()
    pipe.controlnet.requires_grad_(True)


def load_controlnet_pipeline(
    base_model, controlnet_model, device, *, base_revision=None, controlnet_revision=None,
):
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model, revision=controlnet_revision, torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model, revision=base_revision, controlnet=controlnet,
        torch_dtype=dtype, variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    freeze_backbone(pipe)
    pipe.enable_vae_slicing()
    return pipe


def control_image(sketches):
    """Dataset grayscale [-1, 1] -> RGB [0, 1], dark strokes on white paper.

    Preserve pencil shading: no edge detector, threshold, or polarity inversion.
    The same conversion is used during training and sampling.
    """
    return ((sketches.clamp(-1, 1) + 1) / 2).repeat(1, 3, 1, 1)


@torch.no_grad()
def encode_prompts(pipe, prompts):
    tokens = pipe.tokenizer(
        prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    return pipe.text_encoder(tokens.input_ids.to(pipe.device))[0]


def latent_diffusion_loss(pipe, scheduler, sketches, photos, prompt_embeds, *, generator=None):
    """SD 1.5 epsilon-prediction MSE, with sampled VAE latents.

    q_phi(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2)).
    Sampling gives z = mu_phi(x) + sigma_phi(x)*epsilon_enc, epsilon_enc ~ N(0,I).
    The diffusion latent is z_0 = scaling_factor*z.

    Only the VAE encoding is no-grad. Detaching the frozen U-Net forward would
    sever the loss -> U-Net -> ControlNet gradient path.
    """
    with torch.no_grad():
        latents = pipe.vae.encode(photos.to(dtype=pipe.vae.dtype)).latent_dist.sample(generator)
        latents = latents * pipe.vae.config.scaling_factor
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (len(latents),),
        device=latents.device, generator=generator,
    )
    # q(z_t | z_0) = N(sqrt(alpha_bar_t)*z_0, (1-alpha_bar_t)*I).
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    down, mid = pipe.controlnet(
        noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
        controlnet_cond=control_image(sketches), return_dict=False,
    )
    # epsilon_theta(z_t, s, c, t): the frozen U-Net uses ControlNet's learned residuals.
    prediction = pipe.unet(
        noisy_latents, timesteps, encoder_hidden_states=prompt_embeds,
        down_block_additional_residuals=[x.to(noisy_latents.dtype) for x in down],
        mid_block_additional_residual=mid.to(noisy_latents.dtype),
    ).sample
    # Minimize E[(epsilon_theta - epsilon)^2]; there is no VAE reconstruction/KL loss here.
    return F.mse_loss(prediction.float(), noise.float())
