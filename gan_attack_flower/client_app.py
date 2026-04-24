from __future__ import annotations

from logging import INFO
from pathlib import Path

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, ConfigRecord, Context
from flwr.common.logger import log

from .attack import generate_samples, save_comparison, save_grid, train_generator
from .task import (
    CNNClassifier,
    FAKE_CLASS_INDEX,
    Generator,
    evaluate_classifier,
    get_weights,
    load_partition,
    set_weights,
    train_classifier,
)


_GEN_KEY = "adv_generator"
_CFG_KEY = "adv_state"


def _module_to_array_record(model: torch.nn.Module) -> ArrayRecord:
    return ArrayRecord(model.state_dict())


def _array_record_to_module(record: ArrayRecord, model: torch.nn.Module) -> None:
    model.load_state_dict(record.to_torch_state_dict(), strict=True)


def _load_or_init_generator(
    context: Context, latent_dim: int, device: torch.device
) -> tuple[Generator, int]:
    generator = Generator(latent_dim=latent_dim).to(device)
    if _GEN_KEY in context.state:
        _array_record_to_module(context.state[_GEN_KEY], generator)
    round_counter = 0
    if _CFG_KEY in context.state:
        round_counter = int(context.state[_CFG_KEY]["round_counter"])
    return generator, round_counter


def _save_generator(context: Context, generator: Generator, round_counter: int) -> None:
    context.state[_GEN_KEY] = _module_to_array_record(generator)
    context.state[_CFG_KEY] = ConfigRecord({"round_counter": round_counter})


class VictimClient(NumPyClient):
    def __init__(
        self,
        batch_size: int,
        local_epochs: int,
        lr: float,
        lr_decay: float,
        device: torch.device,
    ) -> None:
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.model = CNNClassifier().to(device)
        self.train_loader, self.test_loader = load_partition(
            is_victim=True, batch_size=batch_size
        )

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        loss = train_classifier(
            self.model,
            self.train_loader,
            epochs=self.local_epochs,
            lr=self.lr,
            lr_decay=self.lr_decay,
            device=self.device,
        )
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"train_loss": float(loss), "role": "victim"},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, acc = evaluate_classifier(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}

class AdversaryClient(NumPyClient):
    def __init__(
        self,
        context: Context,
        target_class: int,
        batch_size: int,
        local_epochs: int,
        lr: float,
        lr_decay: float,
        gan_latent_dim: int,
        gan_steps_per_round: int,
        gan_batch_size: int,
        gan_lr: float,
        num_injected_samples: int,
        num_server_rounds: int,
        save_every: int,
        output_dir: str,
        device: torch.device,
    ) -> None:
        self.context = context
        self.device = device
        self.target_class = target_class  # a class the adversary does NOT own
        self.local_epochs = local_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.latent_dim = gan_latent_dim
        self.gan_steps_per_round = gan_steps_per_round
        self.gan_batch_size = gan_batch_size
        self.gan_lr = gan_lr
        self.num_injected_samples = num_injected_samples
        self.num_server_rounds = num_server_rounds
        self.save_every = save_every
        self.output_dir = output_dir

        self.model = CNNClassifier().to(device)
        self.train_loader, self.test_loader = load_partition(
            is_victim=False, batch_size=batch_size
        )

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        generator, round_counter = _load_or_init_generator(
            self.context, self.latent_dim, self.device
        )
        round_counter += 1

        g_loss = train_generator(
            generator=generator,
            discriminator=self.model,
            target_class=self.target_class,
            latent_dim=self.latent_dim,
            steps=self.gan_steps_per_round,
            batch_size=self.gan_batch_size,
            lr=self.gan_lr,
            device=self.device,
        )

        injected_x = generate_samples(
            generator=generator,
            num_samples=self.num_injected_samples,
            latent_dim=self.latent_dim,
            device=self.device,
        ).detach().cpu()
        injected_y = torch.full(
            (self.num_injected_samples,), FAKE_CLASS_INDEX, dtype=torch.long
        )

        set_weights(self.model, parameters)
        loss = train_classifier(
            self.model,
            self.train_loader,
            epochs=self.local_epochs,
            lr=self.lr,
            lr_decay=self.lr_decay,
            device=self.device,
            injected_x=injected_x,
            injected_y=injected_y,
        )

        if round_counter == 1 or round_counter % self.save_every == 0:
            samples = generate_samples(
                generator=generator,
                num_samples=64,
                latent_dim=self.latent_dim,
                device=self.device,
            )
            base = Path(self.output_dir)
            if not base.is_absolute():
                base = Path(__file__).resolve().parent.parent / base
            out_path = base / "reconstructions" / f"round_{round_counter:04d}.png"
            save_grid(samples, out_path, nrow=8)
            log(INFO, "[atacante] salva grid de reconstrução em %s", out_path)

        if round_counter == self.num_server_rounds:
            base = Path(self.output_dir)
            if not base.is_absolute():
                base = Path(__file__).resolve().parent.parent / base
            cmp_path = base / "comparacao_original_vs_reconstrucao.png"
            save_comparison(
                target_class=self.target_class,
                generator=generator,
                latent_dim=self.latent_dim,
                device=self.device,
                path=cmp_path,
                num_samples=10,
            )
            log(INFO, "[atacante] comparação final salva em %s", cmp_path)

        _save_generator(self.context, generator, round_counter)

        return (
            get_weights(self.model),
            len(self.train_loader.dataset) + self.num_injected_samples,
            {
                "train_loss": float(loss),
                "gan_g_loss": float(g_loss),
                "round_counter": int(round_counter),
                "role": "adversary",
            },
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, acc = evaluate_classifier(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


def client_fn(context: Context):
    rc = context.run_config
    nc = context.node_config

    batch_size = int(rc["batch-size"])
    local_epochs = int(rc["local-epochs"])
    lr = float(rc["learning-rate"])
    lr_decay = float(rc["lr-decay"])

    partition_id = int(nc["partition-id"])
    num_partitions = int(nc["num-partitions"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convenção: partition 0 = vítima, partition 1 = atacante.
    if partition_id == 0:
        return VictimClient(
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            lr_decay=lr_decay,
            device=device,
        ).to_client()

    return AdversaryClient(
        context=context,
        target_class=int(rc["target-class"]),
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
        lr_decay=lr_decay,
        gan_latent_dim=int(rc["gan-latent-dim"]),
        gan_steps_per_round=int(rc["gan-steps-per-round"]),
        gan_batch_size=int(rc["gan-batch-size"]),
        gan_lr=float(rc["gan-lr"]),
        num_injected_samples=int(rc["num-injected-samples"]),
        num_server_rounds=int(rc["num-server-rounds"]),
        save_every=int(rc["save-every"]),
        output_dir=str(rc["output-dir"]),
        device=device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
