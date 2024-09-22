import argparse
import copy
import torch
import torch.nn as nn

from src.constants import PAD_CHAR
from src.model import load_model_from_archive
from src.utils import char_from_index, embedding_from_indices, embedding_from_prompt

def generate(device: torch.device, model: nn.Module, prompt: str, sequence_size: int = 16, output_size: int = 100) -> str:
    chars = copy.deepcopy(prompt.split('\s'))

    if len(prompt) < sequence_size:
        pad_size = sequence_size - len(prompt)
        padded_prompt = [PAD_CHAR] * pad_size + list(prompt)
    else:
        padded_prompt = list(prompt[-sequence_size:])

    sequence_embedding = embedding_from_prompt(padded_prompt)
    sequence_embedding = sequence_embedding.unsqueeze(0).to(device)

    model.eval()

    char = None
    i = 0
    while i < (output_size - len(prompt)) and char != '\n':
        with torch.no_grad():
            logits = model(sequence_embedding)
            logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()

        char_idx = torch.multinomial(logits_probs, 1).item()
        char = char_from_index(char_idx)
        chars.append(char)

        char_embedding = embedding_from_indices(torch.tensor([char_idx]))
        char_embedding = char_embedding.unsqueeze(0).to(device)
        sequence_embedding = torch.cat((sequence_embedding[:, -sequence_size+1:, :], char_embedding), dim=1)

        i += 1

    return ''.join(chars)

def prompt(device: torch.device, name: str, text: str, sequence_size: int, output_size: int):
    model = load_model_from_archive(device, name)

    return generate(device, model, text.lower(), sequence_size, output_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text based on a arbitrary corpus.")
    
    parser.add_argument('--name', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--prompt_text', type=str, required=True,
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--output_size', type=int, required=True,
                        help='The size of the generated text output (e.g., "100")')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(prompt(device, args.name, args.prompt_text, args.sequence_size, args.output_size))