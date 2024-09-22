import argparse
import math
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

from src.constants import MODEL_ARCHIVE_DIR, VOCAB_SIZE, PAD_CHAR
from src.corpus_loader import fetch_and_load_corpus
from src.data_loader import create_dataloader
from src.model import model_selector
from src.utils import embedding_from_indices, index_from_char

def train(device: torch.device, corpus: str, name: str, hyperparameters, trial: t.Optional[optuna.Trial] = None, use_tensorboard: bool = False) -> float:
    use_optuna = trial is not None

    tensorboard = None
    if use_tensorboard:
        tensorboard = SummaryWriter()

    dataloader = create_dataloader(
        corpus,
        hyperparameters['sequence_size'],
        hyperparameters['batch_size'],
        hyperparameters['num_workers']
    )

    model: nn.Module = model_selector[name](**{
        "input_size": VOCAB_SIZE,
        "output_size": VOCAB_SIZE,
        "hidden_size": hyperparameters['hidden_size'],
        "cells_size": hyperparameters['cells_size'],
        "dropout": hyperparameters['dropout'],
        "device": device
    })
    
    criterion = nn.CrossEntropyLoss(ignore_index=index_from_char(PAD_CHAR))
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay']
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=hyperparameters['gamma'])

    model_archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR) / model.name
    model_archive_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=VOCAB_SIZE).to(device)
    accumulation_steps = hyperparameters['accumulation_steps']

    model.train()

    start_time = time.time()
    print(f"Training on {device} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(1, hyperparameters['epochs'] + 1)):
        epoch_loss = 0
        epoch_steps = 0

        for step, (sequences, targets) in enumerate(dataloader, 1):
            embedding = embedding_from_indices(indices=sequences).to(device)
            target = targets.to(device)

            logits = model(embedding)

            logits = logits.view(-1, VOCAB_SIZE)  # Flatten logits to shape [batch_size * sequence_size, vocab_size]
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

        torch.save(model, model_archive_dir / f'{epoch}' / f'model.pt')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training on {device} took {total_time:.2f} seconds")

    torch.save(model, model_archive_dir / 'final' / 'model.pt')

    if use_tensorboard:
        tensorboard.flush()
        tensorboard.close()

    return accuracy_metric.compute().item() * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--use_profiler', action='store_true',
                        help="Enable profiling with torch.utils.bottleneck")
    parser.add_argument('--use_tensorboard', action='store_true', 
                        help="Enable tensorboard metrics collection")
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    parser.add_argument('--name', type=str, required=True, choices=['lstm', 'gru'],
                    help='Model to use as a basis for text generation (e.g., "bidirectional")')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--max_steps', type=float, default=+math.inf,
                        help='Number of training steps within the epoch')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for training')
    parser.add_argument('--cells_size', type=int, default=2,
                        help='The number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The size of hidden/context layers')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of samples in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate parameter of LSTM optimizer (Adam is a default setting)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma parameter of LSTM optimizer\'s scheduler (Adam is a default setting) which serves for training stabilization')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay parameter of LSTM optimizer (Adam is a default setting) which serves for weights normalization to avoid overfitting')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Accumulations steps per LSTM optimizer to make backward propagation pass (Adam is a default setting)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = fetch_and_load_corpus(args.url)

    print(f"Training config:\n\tprofiler = {args.use_profiler}\n\ttensorboard = {args.use_tensorboard}\n\tworkers = {args.num_workers}")

    train(
        device,
        corpus,
        name=args.name,
        hyperparameters={
            "num_workers": args.num_workers,
            "max_steps": 100 if args.use_profiler else args.max_steps,
            "epochs": 1 if args.use_profiler else args.epochs,
            "dropout": args.dropout, 
            "cells_size": args.cells_size,
            "hidden_size": args.hidden_size,
            "sequence_size": args.sequence_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "weight_decay": args.weight_decay,
            "accumulation_steps": args.accumulation_steps,
            'steps_patience': 1000,
            'loss_patience': 0.00001
        },
        use_tensorboard=args.use_tensorboard
    )