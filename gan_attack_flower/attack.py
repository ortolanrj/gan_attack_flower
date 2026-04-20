from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from .task import Generator


def train_generator(
    generator: Generator,
    discriminator: nn.Module,
    target_class: int,
    latent_dim: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> float:
    discriminator.eval()
    for p in discriminator.parameters():
        p.requires_grad_(False)

    generator.train()
    optimizer = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.0)
    criterion = nn.NLLLoss()

    target = torch.full((batch_size,), target_class, dtype=torch.long, device=device)

    last_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake = generator(z)
        log_probs = discriminator(fake)
        loss = criterion(log_probs, target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    for p in discriminator.parameters():
        p.requires_grad_(True)

    return last_loss


@torch.no_grad()
def generate_samples(
    generator: Generator,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    generator.eval()
    z = torch.randn(num_samples, latent_dim, device=device)
    return generator(z)


def save_grid(images: torch.Tensor, path: str | os.PathLike, nrow: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = (images.detach().cpu() + 1.0) / 2.0
    grid = grid.clamp(0.0, 1.0)
    save_image(grid, str(path), nrow=nrow)
