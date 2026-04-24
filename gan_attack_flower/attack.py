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


def save_comparison(
    target_class: int,
    generator: Generator,
    latent_dim: int,
    device: torch.device,
    path: str | os.PathLike,
    num_samples: int = 10,
) -> None:
    """Comparação das imagens geradas via GAN, com as imagens reais de treinamento via distância L1.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    targets = torch.tensor(mnist.targets)
    class_idx = (targets == target_class).nonzero(as_tuple=True)[0]
    real_pool = torch.stack([mnist[int(i)][0] for i in class_idx])  # (M, 1, 28, 28)

    num_candidates = max(256, num_samples * 20)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_candidates, latent_dim, device=device)
        all_fakes = generator(z).cpu()

    flat = all_fakes.view(num_candidates, -1)
    selected_idx = [0]
    for _ in range(num_samples - 1):
        sel_flat = flat[selected_idx]
        dists = torch.cdist(flat, sel_flat, p=1).min(dim=1).values
        dists[selected_idx] = -1
        selected_idx.append(int(dists.argmax()))
    fake_images = all_fakes[selected_idx]
    fake_flat = fake_images.view(num_samples, -1)
    real_flat = real_pool.view(real_pool.size(0), -1)

    dists = torch.cdist(fake_flat.float(), real_flat.float(), p=1)
    nearest_idx = dists.argmin(dim=1)
    matched_reals = real_pool[nearest_idx]  # (num_samples, 1, 28, 28)

    matched_reals = ((matched_reals + 1.0) / 2.0).clamp(0.0, 1.0)
    fake_images = ((fake_images + 1.0) / 2.0).clamp(0.0, 1.0)

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 5))
    fig.suptitle(
        f"",
        fontsize=14,
        fontweight="bold",
    )

    axes[0, 0].set_ylabel("Original\nProxima", fontsize=12, fontweight="bold", rotation=0,
                           labelpad=60, va="center")
    axes[1, 0].set_ylabel("Reconstrução\nGAN", fontsize=12, fontweight="bold", rotation=0,
                           labelpad=60, va="center")

    for i in range(num_samples):
        axes[0, i].imshow(
            matched_reals[i].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[0, i].axis("off")
        axes[1, i].imshow(
            fake_images[i].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[1, i].axis("off")

    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
