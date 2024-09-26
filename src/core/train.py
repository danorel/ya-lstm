import optuna
import time
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import typing as t
import pathlib

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.core.constants import MODEL_ARCHIVE_DIR, PAD_TOKEN
from src.core.model import model_selector

# character-level specific imports
from src.character_level.data_loader import create_dataloader as create_character_dataloader
from src.character_level.utils import make_corpus_operations as make_character_operations

# word-level specific imports
from src.word_level.data_loader import create_dataloader as create_word_dataloader
from src.word_level.utils import make_corpus_operations as make_word_operations


def get_config(model_type, corpus: str):
    if model_type == 'character':
        operations = make_character_operations(corpus)
        create_dataloader = create_character_dataloader
    elif model_type == 'word':
        operations = make_word_operations(corpus)
        create_dataloader = create_word_dataloader
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        'vocab_size': operations['vocab_size'],
        'token_to_index': operations['token_to_index'],
        'create_dataloader': create_dataloader,
        'pad_index': operations['token_to_index'][PAD_TOKEN]
    }

def make_trainer(
    create_dataloader,
    token_to_index: dict,
    vocab_size: int,
    pad_index: int,
    trial: t.Optional[optuna.Trial] = None,
    use_tensorboard: bool = False
):
    def train(
        device: torch.device,
        corpus: str,
        model_name: str,
        model_type: str,
        hyperparameters: dict,
    ) -> float:
        use_optuna = trial is not None

        tensorboard = None
        if use_tensorboard:
            tensorboard = SummaryWriter()

        dataloader = create_dataloader(
            device,
            corpus,
            token_to_index,
            hyperparameters['sequence_size'],
            hyperparameters['batch_size'],
            hyperparameters['num_workers']
        )
  
        model_config = {
            "input_size": vocab_size,
            "embedding_size": hyperparameters['embedding_size'],
            "output_size": vocab_size,
            "hidden_size": hyperparameters['hidden_size'],
            "cells_size": hyperparameters['cells_size'],
            "dropout": hyperparameters['dropout'],
            "device": device
        }

        print("Model config:")
        for hyperparameter_name, hyperparamter_value in model_config.items():
            print(f"\t{hyperparameter_name} = {hyperparamter_value}")

        model: nn.Module = model_selector[model_name](**model_config)

        criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay']
        )
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=hyperparameters['gamma'])

        model_archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR) / model_name / model_type
        model_archive_dir.mkdir(parents=True, exist_ok=True)
        
        accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size).to(device)
        accumulation_steps = hyperparameters['accumulation_steps']

        model.train()

        start_time = time.time()
        print(f"Training on {device} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

        best_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(1, hyperparameters['epochs'] + 1)):
            epoch_loss = 0
            epoch_steps = 0

            for step, (sequences, target) in enumerate(dataloader, 1):
                logits = model(sequences)

                logits = logits.view(-1, vocab_size)  # Flatten logits to shape [batch_size * sequence_size, vocab_size]
                target = target.view(-1)  # Flatten target to shape [batch_size * sequence_size]

                assert logits.shape[0] == target.shape[0], f"Shape mismatch: {logits.shape[0]} != {target.shape[0]}"

                loss = criterion(logits, target)
                loss = loss / accumulation_steps

                loss.backward()

                if (step % accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    optimizer.zero_grad()
                
                loss = loss.item()
                epoch_loss += loss

                predictions = torch.argmax(logits, dim=1)
                accuracy_metric.update(predictions, target)
                accuracy = accuracy_metric.compute().item() * 100
                
                if use_tensorboard:
                    tensorboard.add_scalar('Loss/average', (epoch_loss / step), step)
                    tensorboard.add_scalar('Loss/step', loss, step)
                    tensorboard.add_scalar('Accuracy/step', accuracy, step)
                
                if use_optuna:
                    trial.report(accuracy, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                if loss < best_loss - hyperparameters['loss_patience']:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > hyperparameters['steps_patience']:
                    early_stopping_dir = model_archive_dir / f'{epoch}' / 'early_stopping'
                    early_stopping_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Stopping early at epoch {epoch}, step {step}. No improvement in loss for {hyperparameters['steps_patience']} steps.")
                    torch.save(model, early_stopping_dir / 'model.pt')
                    return accuracy_metric.compute().item() * 100
                
                if step % 100 == 0:
                    print(f"Epoch {epoch}/{hyperparameters['epochs']} (step = {step}):\n\tLoss = {loss:.4f}\n\tAccuracy = {accuracy:.4f}")
                    checkpoint_dir = model_archive_dir / f'{epoch}' / f'{step}'
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(model, checkpoint_dir / 'model.pt')

                epoch_steps += 1
                if epoch_steps >= hyperparameters['max_steps']:
                    break

            scheduler.step()

            epoch_accuracy = accuracy_metric.compute().item() * 100

            if use_tensorboard:
                tensorboard.add_scalar('Accuracy/epoch', epoch_accuracy, epoch)
                tensorboard.add_scalar('Loss/epoch', epoch_loss, epoch)
            
            print(f"Epoch {epoch} finished!\n\tTotal Loss: {epoch_loss:.4f}\n\tAccuracy: {epoch_accuracy:.4f}")

            epoch_dir = model_archive_dir / f'{epoch}'
            epoch_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model, epoch_dir / f'model.pt')
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training on {device} took {total_time:.2f} seconds")

        final_dir = model_archive_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, final_dir / 'model.pt')

        if use_tensorboard:
            tensorboard.flush()
            tensorboard.close()

        return accuracy_metric.compute().item() * 100

    return train