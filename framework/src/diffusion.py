from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DiffusionBatch:
    latent: torch.Tensor
    label: Optional[torch.Tensor] = None
    action: Optional[torch.Tensor] = None


class LatentDiffusionAugmentor(nn.Module):
    """
    Placeholder module for diffusion-assisted latent augmentation.

    This is intentionally lightweight for the first framework pass.
    The intended training order is:

    1. stabilize baseline and sequence classifier
    2. extract latent embeddings from the sequence encoder
    3. train diffusion on training-fold latents only
    4. sample conditioned latents for augmentation
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.denoiser = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.denoiser(latent)

    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n_samples, self.latent_dim, device=device)
