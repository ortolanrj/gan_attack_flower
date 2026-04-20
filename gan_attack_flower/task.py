from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


NUM_CLASSES = 11          # 10 dígitos reais + 1 falso para o atacante.
FAKE_CLASS_INDEX = 10
VICTIM_CLASSES = (0, 1, 2, 3, 4)
ADVERSARY_CLASSES = (5, 6, 7, 8, 9)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # Equivalente a nn.SpatialConvolutionMM from Torch7 (Lua)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        self.fc0 = nn.Linear(64 * 7 * 7, 256)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc0(x))
        x = torch.tanh(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


# Gerador DCGAN
class Generator(nn.Module):

    def __init__(self, latent_dim: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.project = nn.Linear(latent_dim, 256 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Inicia pesos da distribuição N(0, 0.02^2), de acordo com a Seção 8.3."""
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, 256, 7, 7)
        return self.main(x)


# Helpers para parâmetros
def get_weights(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


_MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ]
)


def _load_full_mnist(train: bool = True) -> datasets.MNIST:
    return datasets.MNIST(
        root="./data", train=train, download=True, transform=_MNIST_TRANSFORM
    )


# Retorna (train_loader, test_loader) para os participantes
def load_partition(is_victim: bool, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    trainset = _load_full_mnist(train=True)
    testset = _load_full_mnist(train=False)

    train_labels = np.asarray(trainset.targets)
    classes = VICTIM_CLASSES if is_victim else ADVERSARY_CLASSES
    idx = np.where(np.isin(train_labels, classes))[0]

    train_loader = DataLoader(
        Subset(trainset, idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)
    return train_loader, test_loader

# Classificador
def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    lr_decay: float,
    device: torch.device,
    injected_x: torch.Tensor | None = None,
    injected_y: torch.Tensor | None = None,
) -> float:

    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.0, weight_decay=lr_decay
    )
    criterion = nn.NLLLoss()

    running, n_batches = 0.0, 0
    inj_n = int(injected_x.size(0)) if injected_x is not None else 0
    cursor = 0
    per_batch_inj = 0
    if inj_n > 0 and len(loader) > 0:
        per_batch_inj = max(1, inj_n // len(loader))

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if inj_n > 0:
                start = cursor % inj_n
                end = start + per_batch_inj
                if end <= inj_n:
                    ex_x = injected_x[start:end]
                    ex_y = injected_y[start:end]
                else:
                    ex_x = torch.cat([injected_x[start:], injected_x[: end - inj_n]])
                    ex_y = torch.cat([injected_y[start:], injected_y[: end - inj_n]])
                cursor = end % inj_n
                x = torch.cat([x, ex_x.to(device)], dim=0)
                y = torch.cat([y, ex_y.to(device)], dim=0)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

    return running / max(n_batches, 1)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.NLLLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        pred = logits[:, :10].argmax(dim=1)  # ignore fake class at argmax
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total
