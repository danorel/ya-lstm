import argparse
import copy
import torch
import torch.nn as nn
import pathlib

from src.constants import MODELS_DIR
from src.corpus_loader import fetch_and_load_corpus
from src.utils import one_hot_encoding

def load_model(device: str, name: str) -> nn.Module:
    models_dir = pathlib.Path(MODELS_DIR)
    filepath = models_dir / name / 'final_state_dict.pth'
    if not filepath.exists():
        raise RuntimeError("Not found pre-trained LSTM model")
    model: nn.Module = torch.load(filepath)
    model.to(device)
    return model

def generate(device: str, model: nn.Module, prompt: str, char_to_index: dict, index_to_char: dict, vocab_size: int, sequence_size: int = 16, output_size: int = 100) -> str:
    if len(prompt) < sequence_size:
        raise RuntimeError(f"Starting characters should have at least {sequence_size} symbols")
    
    sequence_embedding = one_hot_encoding(
        indices=torch.tensor([char_to_index[char] for char in prompt[-sequence_size:]], dtype=torch.long), 
        vocab_size=vocab_size
    ).unsqueeze(0).to(device)
    chars = copy.deepcopy(prompt.split('\s'))

    model.eval()

    char = None
    i = 0
    while i < (output_size - len(prompt)) and char != '\n':
        with torch.no_grad():
            logits = model(sequence_embedding)
            logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()

        char_idx = torch.multinomial(logits_probs, 1).item()
        char = index_to_char[char_idx]
        chars.append(char)

        char_embedding = one_hot_encoding(torch.tensor([char_idx]), vocab_size).unsqueeze(0).to(device)
        sequence_embedding = torch.cat((sequence_embedding[:, -sequence_size:, :], char_embedding), dim=1)

        i += 1

    return ''.join(chars)

def prompt(device: str, corpus: str, name: str, text: str, sequence_size: int, output_size: int):
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    index_to_char = {idx: char for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    model = load_model(device, name)

    return generate(device, model, text.lower(), char_to_index, index_to_char, vocab_size, sequence_size, output_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text based on a arbitrary corpus.")
    
    parser.add_argument('--name', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the corpus (e.g., Shakespeare corpus: "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")')
    parser.add_argument('--prompt_text', type=str, required=True,
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--output_size', type=int, required=True,
                        help='The size of the generated text output (e.g., "100")')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = fetch_and_load_corpus(args.url)
    
    print(prompt(device, corpus, args.name, args.prompt_text, args.sequence_size, args.output_size))