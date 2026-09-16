# DIO: Refining Mutual Information and Causal Chains to Enhance Machine Abstract Reasoning Ability

Official implementation of **DIO** (Deep-learning Intelligent-model Organized via Causal Chains) and its three refinement methods:

- **Brando** — tightens the variational lower bound on mutual information by constructing learnable hypothetical (incorrect) options;
- **WORLD** — models RPM image features with a trainable Gaussian Mixture Model (GMM), enabling targeted sampling of diverse incorrect options and generative RPM solving;
- **DIEGO** — rectifies the "attributes → patterns" segment of the causal chain via metadata-derived supervision (used during training only; discarded at test time).

---

## 1. Repository Contents

| File | Description |
|---|---|
| `DIO.py` | Solo DIO model (`raven_clip`), including the ℓ_DIO loss (`dio_loss`) |
| `DIO_Brando.py` | DIO + Brando. The Brando network maps Gaussian seeds to constructive hypothetical options; `num_aux_candidates` controls their number |
| `DIO_WORLD.py` | DIO + WORLD (discriminative). GMM modeling of image features with EMA-based component updates |
| `DIO_WORLD_GEN.py` | DIO + WORLD for generative RPM (solution-distribution estimation, Gumbel-Max sampling, decoding) |
| `DIO_WORLD_sn_toxic.py` | DIO + WORLD_G with spectral/Lipschitz normalization and the G-head hierarchical interaction (large-M configuration) |
| `DIO_DIEGO.py` | DIO + DIEGO (metadata-guided causal-chain rectification; also serves as the validator for generative RPM) |
| `Blocks_clip.py` | Shared building blocks: ViT, attention, feed-forward layers, etc. |
| `Infinity_Transformer.py` | Infinite-attention style memory bank used by the Brando network |
| `make_pgm_data.py` | Dataset / DataLoader for PGM-style `.npz` files |

Each model file contains a `__main__` self-test that runs a forward/backward pass on random inputs, prints a `torchinfo` summary, and benchmarks single-instance latency with `torch.utils.benchmark.Timer`.

---

## 2. Environment

| Item | Version / Specification |
|---|---|
| Python | 3.11.7 |
| PyTorch | 2.8.0 (cu129) |
| torchvision | 0.23.0 (cu129) |
| NumPy | 1.26.4 |
| einops | 0.8.1 |
| tqdm | 4.65.0 |
| torchinfo | 1.8.0 |
| Hardware (reference) | 8 × NVIDIA A100 80G |

### Installation

```bash
conda create -n dio python=3.11.7 -y
conda activate dio

# PyTorch 2.8.0 with the CUDA 12.9 build:
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129

# Remaining dependencies
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

Benchmarks: **RAVEN**, **I-RAVEN**, **PGM** (and **Bongard-Logo** for the domain-extension study in the supplementary materials).

Download from the official sources:

- RAVEN: https://github.com/WellyZhang/RAVEN
- I-RAVEN: https://github.com/husheng12345/SRAN
- PGM: https://github.com/google-deepmind/abstract-reasoning-matrices
- Bongard-Logo: https://github.com/NVlabs/Bongard-Logo

The loader in `make_pgm_data.py` expects each problem as an `.npz` file containing:

- `image`: array of shape `(16, 160, 160)` — 8 context images followed by 8 candidate options;
- `target`: integer index of the correct option.

Files must be named so that the split is recoverable from the filename (`..._{train,val,test}_...`), and placed under per-task folders, e.g.:

```
./neutral/            # PGM-Neutral
./interpolation/      # PGM-Interpolation
./extrapolation/      # PGM-Extrapolation
...
```

**Preprocessing**: images are resized from 160×160 to **80×80** with bilinear interpolation before entering the model; the ViT uses a 20×20 patch size, i.e. **N = 16** patches per image.

---

## 4. Quick Start (Self-Tests)

Every model file ships with a smoke test:

```bash
python DIO.py                  # solo DIO: forward/backward + parameter summary + latency
python DIO_Brando.py           # DIO + Brando
python DIO_WORLD.py            # DIO + WORLD 
python DIO_WORLD_GEN.py        # DIO + WORLD_G 
python DIO_DIEGO.py            # DIO + DIEGO
```

---

## 5. Using the Models in Training

All models expose the same interface: a forward pass over a batch of 16 images per instance, and a `loss_function` returning the objective and accuracy statistics.

```python
import torch
from DIO_Brando import raven_clip   # or: from DIO import raven_clip, etc.

model = raven_clip(num_aux_candidates=8).cuda()   # 8 hypothetical options <=> beta = 24 in the paper
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

for images, _, _, idx in train_loader:            # images: (B, 16, 80, 80)
    images, idx = images.cuda(), idx.long().cuda()
    out = model(images)
    loss, right_shape, right_line, right = model.loss_function(
        *out, target_shape=None, target_line=None, idx=idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
scheduler.step()
```

**Training configuration** (identical across configurations, per Table XVI of the paper): AdamW with learning rate 1e-3; StepLR (step size 1 epoch, γ = 0.99); batch size 100 per GPU on 8 × A100; base random seed 2048 with the GPU rank added as an offset (`seed = 2048 + local_rank`) so each process starts from a distinct random state while the experiment remains deterministic.

### Configuration ↔ Paper Correspondence

| Paper configuration | Code entry point | Key settings |
|---|---|---|
| Solo DIO | `DIO.py : raven_clip()` | IFEM 128/3/4, PPIM & PCEM 64/3/4 (dim/depth/heads) |
| DIO + Brando | `DIO_Brando.py : raven_clip(num_aux_candidates=8)` | 8 hypothetical options (β = 24); Brando latent dim 64, depth L = 9, H = 4 heads, head dim 16; memory matrix init. 0, buffer init. 1e-9 |
| DIO + WORLD | `DIO_WORLD.py` | M = 2^11 GMM components; decoder 64/7/4; 256 samples (16 per token) |
| DIO + WORLD_G | `DIO_WORLD_GEN.py` | M = 2^14 components, G = 2 heads (hierarchical interaction); spectral/Lipschitz normalization |
| DIO + DIEGO | `DIO_DIEGO.py` | scaling coefficient τ init. 1e-6; reference vector dim 64 |

Notes:

- **DIO + Brando** requires a pre-trained DIO checkpoint: DIO is first trained with ℓ_DIO until near-optimal, then Brando is integrated and the loss switches to ℓ_Brando; during joint training DIO's parameters (except the image feature extraction module) are frozen every other iteration.
- **WORLD components** are initialized from N(0, 1) and ℓ2-normalized; the EMA decay η is 0.99 during active training and switches to 0.9999 after the backbone is frozen (EMA ε = 1e-5); loss coefficients in EMA mode: ℓ1 : ℓ2 : ℓ4 : ℓ5 = 6 : 1 : 20 : 20. Auxiliary negatives are sampled from the top-50% most-used components, optionally with an inter-instance strategy (an engineering approximation, not a theoretical equivalence).
- **DIEGO** uses metadata only during training; the reference dictionary is discarded at test time.




