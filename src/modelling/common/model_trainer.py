from dataclasses import dataclass
import pathlib
import time
import typing as t

import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm import tqdm

from src.constants.corpus import PAD_TOKEN
from src.constants.device import device
from src.constants.metadata import MODEL_ARCHIVE_DIR, TENSORBOARD_DIR
from src.modelling.common.model_arguments import CorpusUtils, Metadata
from src.modelling.common.model_loader import select_architecture


@dataclass
class Architecture:
    embedding_size: int
    hidden_size: int
    cells_size: int
    dropout: float
    input_size: t.Optional[int] = None
    output_size: t.Optional[int] = None


@dataclass
class Hyperparameters:
    sequence_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float
    grad_clip_norm: float


@dataclass
class Setup:
    max_epochs: int
    max_steps: int
    accumulation_steps: t.Optional[int] = 1
    patience_steps: t.Optional[int] = 100
    log_steps: t.Optional[int] = 10


class Plugins:
    def __init__(self, use_tensorboard: bool):
        self.archive = None
        self.tensorboard = SummaryWriter(log_dir=TENSORBOARD_DIR) if use_tensorboard else None

    def setup_archive_from(self, metadata: Metadata):
        self.archive = pathlib.Path(MODEL_ARCHIVE_DIR) / metadata.architecture_name / metadata.modelling_name
        self.archive.mkdir(parents=True, exist_ok=True)


def print_config(config, description: str):
    print(description)
    for name, value in config.__dict__.items():
        print(f"\t{name} = {value}")


def measure_time(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Training on {device} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        output = f(*args, **kwargs)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training on {device} took {total_time:.2f} seconds")

        return output

    return wrapper


def find_accuracy(metric: torchmetrics.Accuracy, use_percents: bool = True):
    accuracy_in_units = metric.compute().item()
    if not use_percents:
        return accuracy_in_units
    accuracy_in_percents = accuracy_in_units * 100
    return accuracy_in_percents


def find_stability(model: nn.Module):
    grad_norms = [param.grad.norm(2).item() for _, param in model.named_parameters() if param.grad is not None]
    return np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)


def find_class_weights(dataloader, vocab_size: int):
    class_counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for _, target in dataloader:
        target = target.view(-1)
        unique, counts = torch.unique(target, return_counts=True)
        class_counts[unique] += counts
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = torch.pow(class_weights, 0.5)
    class_weights = class_weights / class_weights.sum()
    return class_weights


def make_optimizer_and_scheduler(model: nn.Module, hyperparameters: Hyperparameters) -> tuple:
    optimizer = optim.Adam(
        model.parameters(), lr=hyperparameters.learning_rate, weight_decay=hyperparameters.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparameters.gamma)
    return optimizer, scheduler


def save_model(model: nn.Module, archive: pathlib.Path, path: str) -> None:
    model_checkpoint_dir = archive / path
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_checkpoint_dir / "model.pt")


def make_trainer(metadata: Metadata, plugins: Plugins, setup: Setup, corpus_utils: CorpusUtils):
    """Return a trainer function based on provided configurations."""
    plugins.setup_archive_from(metadata)

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=corpus_utils.vocab_size).to(device)

    def setup_architecture(architecture: Architecture):
        return Architecture(
            input_size=corpus_utils.vocab_size,
            embedding_size=architecture.embedding_size,
            hidden_size=architecture.hidden_size,
            cells_size=architecture.cells_size,
            dropout=architecture.dropout,
            output_size=corpus_utils.vocab_size,
        )

    def train_one_epoch(
        epoch: int,
        progress_step: int,
        model: nn.Module,
        dataloader,
        optimizer,
        criterion,
        hyperparameters: Hyperparameters,
    ):
        """Train the model for one epoch."""
        model.train()

        epoch_total_loss = 0
        epoch_best_loss = float("+inf")
        stale_steps = 0

        for step, (indices, target) in enumerate(dataloader, 1):
            if step > setup.max_steps:
                print(f"Stopping early at step {step}. Reached max steps: {setup.max_steps}")
                break

            logits = model(indices)

            logits = logits.view(-1, corpus_utils.vocab_size)
            target = target.view(-1)

            batch_loss = criterion(logits, target)
            batch_loss /= setup.accumulation_steps

            batch_loss.backward()

            batch_loss = batch_loss.item()
            epoch_total_loss += batch_loss
            epoch_mean_loss = epoch_total_loss / step

            prediction = torch.argmax(logits, dim=1)
            metric.update(prediction, target)

            if batch_loss < epoch_best_loss:
                epoch_best_loss = batch_loss
                stale_steps = 0
            else:
                stale_steps += 1

            if stale_steps > setup.patience_steps:
                save_model(model, plugins.archive, path=f"{epoch}/early_stopping")
                print(f"Stopping at step {step}. No improvement in loss for {setup.patience_steps} steps.")
                break

            if step % setup.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

            if step % setup.log_steps == 0:
                save_model(model, plugins.archive, path=f"{epoch}/{step}")
                save_model(model, plugins.archive, path="latest")
                print(f"Step {step} Loss: {epoch_mean_loss:.4f}, Accuracy: {find_accuracy(metric):.4f}%")

            if plugins.tensorboard:
                plugins.tensorboard.add_scalar("Step/Average-Loss", epoch_mean_loss, progress_step)
                plugins.tensorboard.add_scalar("Step/Total-Loss", epoch_total_loss, progress_step)
                plugins.tensorboard.add_scalar("Step/Batch-Loss", batch_loss, progress_step)
                plugins.tensorboard.add_scalar("Step/Accuracy", find_accuracy(metric), progress_step)
                plugins.tensorboard.add_scalar("Step/Stability", find_stability(model), progress_step)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        plugins.tensorboard.add_histogram(f"Step/Grad/{name}", param.grad, progress_step)
                        plugins.tensorboard.add_scalar(
                            f"Step/Grad-Norm/{name}", param.grad.norm(2).item(), progress_step
                        )

            progress_step += 1

        return epoch_total_loss, find_accuracy(metric)

    @measure_time
    def train(architecture: Architecture, hyperparameters: Hyperparameters) -> float:
        architecture = setup_architecture(architecture)
        print_config(plugins, description="Settings:")
        print_config(setup, description="Setup:")
        print_config(architecture, description="Architecture:")
        print_config(hyperparameters, description="Hyperparameters:")

        model = select_architecture[metadata.architecture_name](**architecture.__dict__)
        dataloader = corpus_utils.create_dataloader(hyperparameters.sequence_size, hyperparameters.batch_size)
        criterion = nn.CrossEntropyLoss(
            ignore_index=corpus_utils.token_to_index[PAD_TOKEN],
            weight=find_class_weights(dataloader, corpus_utils.vocab_size),
        )
        optimizer, scheduler = make_optimizer_and_scheduler(model, hyperparameters)

        progress_step = 0

        for epoch in tqdm(range(1, setup.max_epochs + 1)):
            epoch_loss, epoch_accuracy = train_one_epoch(
                epoch, progress_step, model, dataloader, optimizer, criterion, hyperparameters
            )

            metric.reset()
            scheduler.step()

            if plugins.tensorboard:
                plugins.tensorboard.add_scalar("Epoch/Accuracy", epoch_accuracy, epoch)
                plugins.tensorboard.add_scalar("Epoch/Total-Loss", epoch_loss, epoch)

            save_model(model, plugins.archive, path=f"{epoch}")
            print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%")

            progress_step += setup.max_epochs

        if plugins.tensorboard:
            plugins.tensorboard.flush()
            plugins.tensorboard.close()

        return find_accuracy(metric)

    return train
