# Face Sketch Diffusion

A learning project for sketch-conditioned face generation with diffusion.

The goal is to keep the math visible: the reusable diffusion equations and U-Net live in small Python files, while the training loop and visual checks live in notebooks.

## Structure

- `models/diffusion.py` - cosine/linear schedules, forward noising, v-prediction targets, DDPM and DDIM sampling.
- `models/unet.py` - sketch-conditioned U-Net with timestep embeddings, GroupNorm residual blocks, sketch feature injection, and bottleneck attention.
- `models/unet_v2.py` - stronger denoiser with adaptive GroupNorm, deeper residual stages, residual resampling, two attention resolutions, and direct decoder conditioning. The original U-Net stays unchanged for old checkpoints.
- `utils/data.py` - paired manifest dataset with shared sketch/photo crops, affine transforms, and flips.
- `utils/visualization.py` - compact notebook visualizations.
- `scripts/prepare_data.py` - scans raw downloaded datasets and writes `data/processed/pairs.csv`.
- `notebooks/01_train_diffusion.ipynb` - original architecture with new augmentation, saved separately from the completed baseline.
- `notebooks/02_explore_trained_model.ipynb` - executed UNetV2 exploration: forward noising, reverse-diffusion animation, seed variation, fixed-noise sketch changes, and correct/shuffled/blank conditioning checks.
- `notebooks/03_train_unet_v2.ipynb` - V2 training from scratch, fixed-noise EMA validation, and images across training epochs.
- `models/pretrained_controlnet.py` - frozen Stable Diffusion backbone, sketch conditioning, and latent denoising objective.
- `notebooks/04_pretrained_controlnet.ipynb` - pretrained generation, ControlNet-only fine-tuning, matched before/after panels, and latent diffusion frames.
- `notebooks/05_compare_approaches.ipynb` - executed comparison of UNetV2 and SD 1.5 + Line-Art ControlNet, with the ControlNet's published versus face-pair fine-tuned weights clearly distinguished.

### Model names in the figures

The comparison in notebook 05 contains two model architectures and three weight states. Notebook 01 trains the original, smaller U-Net separately.

| Model | Weight state | Trained on our pairs |
| --- | --- | --- |
| UNetV2 | Epoch-12 EMA checkpoint, initialized randomly | Entire model |
| Stable Diffusion 1.5 + Line-Art ControlNet | Published SD 1.5 and `lllyasviel/control_v11p_sd15_lineart` weights | Nothing |
| Stable Diffusion 1.5 + Line-Art ControlNet | Same frozen SD 1.5, face-pair fine-tuned ControlNet | Only ControlNet, 1,000 updates |

Both ControlNet variants start with pretrained weights. In the adapted variant,
the VAE, text encoder and SD U-Net remain frozen. "Fine-tuned" describes the
ControlNet's weight state, not a separate architecture.

Notebook 02 uses only UNetV2 and does not train it. Its four validation sketches,
noise seeds and 60-step DDIM settings are fixed. Blank conditioning means a white
image, not an unconditional model. Notebook 05 directly loads the completed comparison
generations; its newly labeled panels go to `outputs/model_comparison/`, leaving
the original run artifacts intact. Figures, animations and numeric summaries are
embedded in the notebooks for inspection without downloading the data or weights.
Notebook 01 was executed end to end on October 1, 2026. The training outputs in
notebooks 03 and 04 come from the completed September 30 runs; their plots and the
analyses in 02 and 05 were rendered again from those saved results. Rerunning the
cells requires the local data and model artifacts.

### Notebooks

1. [Original U-Net: diffusion and training](notebooks/01_train_diffusion.ipynb)
2. [UNetV2: sampling dynamics and conditioning](notebooks/02_explore_trained_model.ipynb)
3. [UNetV2: architecture and training](notebooks/03_train_unet_v2.ipynb)
4. [Stable Diffusion 1.5: Line-Art ControlNet adaptation](notebooks/04_pretrained_controlnet.ipynb)
5. [Matched model comparison](notebooks/05_compare_approaches.ipynb)

Open the notebooks with their kernel working directory set to `notebooks/`.
Paths use its parent as the project root. Run cells top to bottom: training starts
from fresh model initialization in 01/03 and published weights in 04, with no
automatic resume or alternate data paths. Notebook 02 generates from the trained
UNetV2; notebook 05 analyzes the saved 16-pair results without running inference.

## Data

Training applies the same translation (up to 10% per axis), rotation (up to 12 degrees), scale (0.9-1.1), random crop, and horizontal flip to each sketch/photo pair. Geometry is applied before cropping, with edge padding instead of black borders or mirrored facial features. Validation and test previews use deterministic resizing without augmentation.

Put raw datasets under `data/raw` or use the same remote layout:

```bash
python scripts/prepare_data.py --raw-root data/raw --out data/processed/pairs.csv
```

Every notebook uses this project's `data/processed/pairs.csv`.

The complete local master copy is under `data/raw/`, including the original downloads in `data/raw/archives/`. It uses about 3.41 GB: 1.80 GB extracted and 1.61 GB in archives. Both `data/raw/` and `data/processed/` are gitignored. The local manifest is `data/processed/pairs.csv`; it preserves the original 21,913 train, 1,000 validation, and 1,725 test pairs. Training and model-selection previews use the explicit `val` split, not test images.

When moving to Runpod or another machine, copy `data/raw/` and run the preparation command above from the repository root. The CSV contains absolute paths, so regenerate it at the destination. No AWS access is needed for the data. `data/processed/local_data_receipt.json` records download sources, archive hashes, verification counts, and local sizes. The original folders outside this repository were left unchanged.

The 100 paired examples in `face_sketch` and `cufs_kaggle` are byte-identical (both sketch and photo SHA-256 match). Both labels are retained to reproduce the old baseline manifest, not counted as independent evidence. A deduplicated training experiment should exclude one of these mirrors explicitly.

## Experiments

| Run | Architecture | Training augmentation | Checkpoint |
| --- | --- | --- | --- |
| A: completed September 5 baseline | UNet, 22.78M | crop and flip | `diffusion_sketch_faces.pt` |
| B: completed October 1, 12 epochs | same UNet | crop, flip, affine | `diffusion_sketch_faces_aug.pt` |
| C: completed September 30, 12 epochs | UNetV2, 41.48M | same as B | `diffusion_sketch_faces_v2.pt` |

The original checkpoint remains on the stopped AWS instance's EBS disk; only its executed notebooks and figures were downloaded locally. No new run overwrites that checkpoint. Retrieve it before generating a controlled comparison panel.

- A vs C measures the combined recipe improvement. Augmentation does not invalidate A as a baseline.
- A vs B estimates the effect of augmentation; B vs C compares the architecture package. Single runs still include seed variation. Set `affine=False` in a training loader to recover the original crop/flip distribution; `random_transform=False` would remove both and is not the same baseline.
- Keep data splits, batch size, 12 epochs (32,868 updates on the full manifest), objective, and 60-step deterministic DDIM fixed initially. Equal updates are not equal runtime: report GPU time as well.
- Notebook 02 provides a fixed-sketch, fixed-noise protocol for UNetV2. Comparing the original U-Net requires loading its architecture and checkpoint under the same protocol. Existing screenshots with different sampling noise are illustrative, not a controlled evaluation.
- V2's 128-pair fixed-noise validation loss is a denoising diagnostic, not a measure of realism. Next evaluate a larger shared validation panel for perceptual quality, artifacts, sketch adherence, and sample diversity. Report paired perceptual metrics and distributional quality separately; there is not a unique correct photo for a sketch.
- Check cross-dataset duplicate photos and identity overlap before claiming unseen-identity generalization. The old run's previews already included test images, so that historical inspection cannot be called untouched testing.

Runs B and C are complete. Run B executed 32,868 updates and the final sampling panel in about 43.5 minutes on an RTX 4090. Its checkpoint is saved locally, separately from A and C. Notebook 02 explores C's epoch-12 EMA checkpoint, and notebook 05 compares C with the two SD 1.5 + Line-Art ControlNet weight states. A matched evaluation of A, B and C has not been performed, so the effect of augmentation versus architecture has not been isolated. Changing architecture requires fresh weights; the A checkpoint cannot initialize V2 directly.

V2 uses design ideas from [ADM](https://arxiv.org/abs/2105.05233) and [Palette](https://arxiv.org/abs/2111.05826), not their complete architectures or pretrained weights. The cosine schedule, v-target, EMA decay, learning rate, and reconstruction-loss weight remain unchanged. Architectural plausibility alone is not evidence of better generated faces; that requires training and evaluation.

## Separate pretrained approach

Notebook 04 starts with [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and the author's [line-art ControlNet](https://huggingface.co/lllyasviel/control_v11p_sd15_lineart). Both repositories were publicly accessible without an access request when checked. Exact weight revisions are passed in the notebook. Read their OpenRAIL license restrictions before use.

```bash
pip install -r requirements-pretrained.txt
```

The from-scratch notebooks do not need these extra dependencies. Model loading in notebook 04 downloads several GB into the normal Hugging Face cache. The 1,000-update ControlNet adaptation pilot completed on an L40S using single-image microbatches, accumulation over eight examples and activation checkpointing. Its full-model preflight measured 9.075 GiB peak allocation; training with periodic diagnostics took about 20.8 minutes. The adapted ControlNet, training state and comparison outputs are backed up locally; the frozen base model can be downloaded again from its pinned revision.

The experiment has three stages:

1. Generate 16 validation faces with the existing pretrained ControlNet, before any tuning on our pairs.
2. Freeze the VAE, text encoder, and base U-Net; fine-tune only the already-pretrained ControlNet. Start with 1,000 updates, effective batch eight, learning rate 1e-5, and validation every 250 updates. This is an 8,000-example pilot, not an assumed sufficient training budget.
3. Generate the same validation faces with identical prompts, seeds, DDIM steps, and control strength. Notebook 05 adds V2 outputs and a clearly labeled pixel-error diagnostic at a common 256x256 size.

Sketches remain dark strokes on white backgrounds, converted to RGB [0,1]; photos enter the VAE in [-1,1]. No target photograph or target-derived edge map is used during generation. The fixed generic portrait prompt avoids per-person caption leakage. SD 1.5 uses its pretrained latent noise schedule and epsilon target, not V2's pixel-space cosine/v-prediction objective. Keep the two losses separate in the analysis.

Outputs live under `outputs/pretrained_lineart/`; tuned ControlNet weights live under `checkpoints/pretrained_lineart/controlnet/`. Notebook 04 saves `before.pt` before training and `comparison.pt` after training, with samples, pair identifiers, sampling settings, revisions, and diagnostic histories. It runs the explicit 1,000-update experiment and saves the adapted ControlNet at the end. The archived training state from the completed cloud run is not loaded by the notebook. Existing scratch checkpoints are untouched.

The pretrained method has much more external training data and model capacity, plus 512x512 generation and text guidance. Compare practical quality and additional GPU time, not raw losses or supposedly equal training budgets. Pixel MAE can favor blur and cannot establish realism or identity; inspect artifacts, sketch adherence, and seed diversity separately. The small validation panel is preparation for analysis, not a benchmark or evidence of improvement.
