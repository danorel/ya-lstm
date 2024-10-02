import numpy as np
import time
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import typing as t
import pathlib

from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.core.constants import MODEL_ARCHIVE_DIR, PAD_TOKEN
from src.core.model import model_selector

from src.character_level.utils import make_corpus_operations as make_character_utils
from src.word_level.utils import make_corpus_operations as make_word_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArchitecture:
    embedding_size: int
    hidden_size: int
    cells_size: int
    dropout: float
    input_size: t.Optional[int] = None
    output_size: t.Optional[int] = None

@dataclass
class ModelTrainingConfig:
    max_epochs: int
    max_steps: int
    accumulation_steps: t.Optional[int] = 1
    patience_steps: t.Optional[int] = 100
    log_steps: t.Optional[int] = 10

@dataclass
class ModelHyperparameters:
    sequence_size: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float
    grad_clip_norm: float


def print_config(config, description: str):
    print(description)
    for name, value in config.__dict__.items():
        print(f"\t{name} = {value}")

def get_training_utils(model_type: str, corpus: str):
    if model_type == 'character':
        utils = make_character_utils(corpus)
    elif model_type == 'word':
        utils = make_word_utils(corpus)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return utils

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
    grad_norms = [
        param.grad.norm(2).item()
        for _, param in model.named_parameters()
        if param.grad is not None
    ]
    return np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)

def find_class_weights(dataloader, vocab_size: int):
    class_counts = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for _, target in dataloader:
        target = target.view(-1)
        unique, counts = torch.unique(target, return_counts=True)
        class_counts[unique] += counts
    total_samples = class_counts.sum().item()
    class_weights = total_samples / (class_counts + 1e-6)
    class_weights /= class_weights.max()
    return class_weights

def make_optimizer_and_scheduler(model: nn.Module, hyperparameters: ModelHyperparameters) -> tuple:
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparameters.gamma)
    return optimizer, scheduler

def save_model(model: nn.Module, directory: pathlib.Path, path: str) -> None:
    checkpoint_dir = directory / path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, checkpoint_dir / 'model.pt')

def make_trainer(
    model_name: str,
    model_type: str,
    vocab_size: int,
    token_to_index: dict,
    use_dataloader,
    training_config: ModelTrainingConfig,
    use_tensorboard: bool = False,
    **kwargs
):
    """Return a trainer function based on provided configurations."""
    tensorboard = SummaryWriter() if use_tensorboard else None

    metric = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size).to(device)

    archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR) / model_name / model_type
    archive_dir.mkdir(parents=True, exist_ok=True)

    def setup_architecture(hidden_architecture: ModelArchitecture):
        return ModelArchitecture(
            input_size=vocab_size,
            embedding_size=hidden_architecture.embedding_size,
            hidden_size=hidden_architecture.hidden_size,
            cells_size=hidden_architecture.cells_size,
            dropout=hidden_architecture.dropout,
            output_size=vocab_size
        )

    def train_one_epoch(
        epoch: int,
        progress_step: int,
        model: nn.Module,
        dataloader,
        optimizer,
        criterion,
        hyperparameters: ModelHyperparameters,
    ):
        """Train the model for one epoch."""
        model.train()

        epoch_total_loss = 0
        epoch_best_loss = float('+inf')
        staleness_steps = 0
        
        for step, (indices, target) in enumerate(dataloader, 1):
            logits = model(indices)

            logits = logits.view(-1, vocab_size)
            target = target.view(-1)

            batch_loss = criterion(logits, target)
            batch_loss /= training_config.accumulation_steps

            batch_loss.backward()

            batch_loss = batch_loss.item()
            epoch_total_loss += batch_loss
            epoch_mean_loss = (epoch_total_loss / step)

            prediction = torch.argmax(logits, dim=1)
            metric.update(prediction, target)

            if batch_loss < epoch_best_loss:
                epoch_best_loss = batch_loss
                staleness_steps = 0
            else:
                staleness_steps += 1

            if staleness_steps > training_config.patience_steps:
                save_model(model, archive_dir, path=f'{epoch}/early_stopping')
                print(f"Stopping early at step {step}. No improvement in loss for {training_config.patience_steps} steps.")
                break 

            if step % training_config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparameters.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            if step % training_config.log_steps == 0:
                save_model(model, archive_dir, path=f'{epoch}/{step}')
                print(f"Step {step} Loss: {epoch_mean_loss:.4f}, Accuracy: {find_accuracy(metric):.4f}%")

            if use_tensorboard:
                tensorboard.add_scalar('Step/Average-Loss', epoch_mean_loss, progress_step)
                tensorboard.add_scalar('Step/Total-Loss', epoch_total_loss, progress_step)
                tensorboard.add_scalar('Step/Batch-Loss', batch_loss, progress_step)
                tensorboard.add_scalar('Step/Accuracy', find_accuracy(metric), progress_step)
                tensorboard.add_scalar('Step/Stability', find_stability(model), progress_step)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        tensorboard.add_histogram(f"Step/Grad/{name}", param.grad, progress_step)
                        tensorboard.add_scalar(f"Step/Grad-Norm/{name}", param.grad.norm(2).item(), progress_step)

            progress_step += 1

        return epoch_total_loss, find_accuracy(metric)

    @measure_time
    def train(hidden_architecture: ModelArchitecture, hyperparameters: ModelHyperparameters) -> float:
        architecture = setup_architecture(hidden_architecture)
        print_config(training_config, description="Model training config:")
        print_config(architecture, description="Model architecture:")
        print_config(hyperparameters, description="Model hyperparameters:")

        dataloader = use_dataloader(hyperparameters.sequence_size, hyperparameters.batch_size)
        model = model_selector[model_name](**architecture.__dict__)
        criterion = nn.CrossEntropyLoss(ignore_index=token_to_index[PAD_TOKEN], weight=find_class_weights(dataloader, vocab_size))
        optimizer, scheduler = make_optimizer_and_scheduler(model, hyperparameters)

        progress_step = 0

        for epoch in tqdm(range(1, training_config.max_epochs + 1)):
            epoch_loss, epoch_accuracy = train_one_epoch(epoch, progress_step, model, dataloader, optimizer, criterion, hyperparameters)

            metric.reset()
            scheduler.step()

            if tensorboard:
                tensorboard.add_scalar('Epoch/Accuracy', epoch_accuracy, epoch)
                tensorboard.add_scalar('Epoch/Total-Loss', epoch_loss, epoch)
            
            save_model(model, archive_dir, path=f'{epoch}')
            print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}%")

            progress_step += training_config.max_epochs
        
        if tensorboard:
            tensorboard.flush()
            tensorboard.close()

        return find_accuracy(metric)
    
    return train