import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib

from tqdm import tqdm

from src.constants import MODELS_DIR
from src.corpus_loader import fetch_and_load_corpus
from src.data_loader import dataloader
from src.model import model_selector

def train(corpus: str, name: str, epochs: int, dropout: float, sequence_size: int, batch_size: int, learning_rate: float, weight_decay: float):
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    hidden_size = vocab_size * 4
    
    hyperparameters = {
        "input_size": vocab_size,
        "hidden_size": hidden_size,
        "output_size": vocab_size,
        "dropout": dropout
    }
    model: nn.Module = model_selector[name](**hyperparameters)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_dir = pathlib.Path(MODELS_DIR) / model.name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0

        context = model.init_hidden_state(batch_size)
        hidden_state = model.init_hidden_state(batch_size)

        for batch, (embedding, target) in enumerate(dataloader(corpus, char_to_index, vocab_size, sequence_size, batch_size)):
            optimizer.zero_grad()
            
            logits, context, hidden_state = model(embedding, context, hidden_state)

            hidden_state = hidden_state.detach()
            context = context.detach()

            loss = criterion(logits.view(-1, vocab_size), target.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss = loss.item()
            total_loss += loss

            if batch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch Loss (batch = {batch}): {loss:.4f}")
        
        print(f"Epoch {epoch}/{epochs}, Total Loss: {total_loss:.4f}")
        torch.save(model, model_dir / f'{epoch}_state_dict.pth')

    torch.save(model, model_dir / 'final_state_dict.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--name', type=str, required=True, choices=['lstm', 'gru'],
                    help='Model to use as a basis for text generation (e.g., "bidirectional")')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for training')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of samples in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate parameter of LSTM optimizer (Adam is a default setting)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay parameter of LSTM optimizer (Adam is a default setting) which serves for weights normalization to avoid overfitting')
    
    args = parser.parse_args()

    corpus = fetch_and_load_corpus(args.url)
    
    train(corpus, args.name, args.epochs, args.dropout, args.sequence_size, args.batch_size, args.learning_rate, args.weight_decay)