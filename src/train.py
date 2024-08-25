import argparse
import math
import optuna
import time
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import typing as t
import pathlib

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.constants import MODELS_DIR
from src.corpus_loader import fetch_and_load_corpus
from src.data_loader import dataloader
from src.model import model_selector

def train(device: torch.device, corpus: str, name: str, hyperparameters, trial: t.Optional[optuna.Trial] = None) -> float:
    tensorboard = SummaryWriter()

    vocab = sorted(set(corpus))
    vocab_size = len(vocab)
    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    
    model: nn.Module = model_selector[name](**{
        "input_size": vocab_size,
        "output_size": vocab_size,
        "hidden_size": hyperparameters['hidden_size'],
        "lstm_size": hyperparameters['lstm_size'],
        "dropout": hyperparameters['dropout'],
        "device": device
    })
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparameters['learning_rate'],
        weight_decay=hyperparameters['weight_decay']
    )

    model_dir = pathlib.Path(MODELS_DIR) / model.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size).to(device)
    
    model.train()

    start_time = time.time()
    print(f"Training on {device} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    for epoch in tqdm(range(1, hyperparameters['epochs'] + 1)):
        epoch_loss = 0
        epoch_steps = 0

        for step, (embedding, target) in enumerate(dataloader(device, corpus, char_to_index, vocab_size, hyperparameters['sequence_size'], hyperparameters['batch_size']), 1):
            embedding, target = embedding.to(device), target.to(device)

            logits = model(embedding)

            logits = logits.view(-1, vocab_size)  # Flatten logits to shape [batch_size * sequence_size, vocab_size]
            target = target.view(-1)  # Flatten target to shape [batch_size * sequence_size]

            assert logits.shape[0] == target.shape[0], f"Shape mismatch: {logits.shape[0]} != {target.shape[0]}"

            loss = criterion(logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss = loss.item()
            epoch_loss += loss
            
            predictions = torch.argmax(logits, dim=1)
            accuracy_metric.update(predictions, target)
            accuracy = accuracy_metric.compute().item() * 100

            tensorboard.add_scalar('Loss/average', (epoch_loss / step), step)
            tensorboard.add_scalar('Loss/step', loss, step)
            tensorboard.add_scalar('Accuracy/step', accuracy, step)
            
            if trial is not None:
                trial.report(accuracy, epoch)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}/{hyperparameters['epochs']} (step = {step}): Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

            epoch_steps += 1
            if epoch_steps >= hyperparameters['max_steps']:
                break

        epoch_accuracy = accuracy_metric.compute().item() * 100

        tensorboard.add_scalar('Accuracy/epoch', epoch_accuracy, epoch)
        tensorboard.add_scalar('Loss/epoch', epoch_loss, epoch)
        
        print(f"Epoch {epoch} finished with Total Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        torch.save(model, model_dir / f'{epoch}_state_dict.pth')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training on {device} took {total_time:.2f} seconds")

    torch.save(model, model_dir / 'final_state_dict.pth')

    tensorboard.flush()
    tensorboard.close()

    return accuracy_metric.compute().item() * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
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
    parser.add_argument('--lstm_size', type=int, default=2,
                        help='The number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The size of hidden/context layers')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of samples in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate parameter of LSTM optimizer (Adam is a default setting)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay parameter of LSTM optimizer (Adam is a default setting) which serves for weights normalization to avoid overfitting')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = fetch_and_load_corpus(args.url)
    
    train(device, corpus, name=args.name, hyperparameters={
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "dropout": args.dropout, 
        "lstm_size": args.lstm_size,
        "hidden_size": args.hidden_size,
        "sequence_size": args.sequence_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay
    })