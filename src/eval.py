import argparse
import torch
import torch.nn as nn

from src.core.constants import UNKNOWN_TOKEN, END_OF_THOUGHT_TOKEN
from src.core.corpus_loader import fetch_and_load_corpus
from src.core.model import load_model_from_archive

# character-level specific imports
from src.character_level.utils import (
    input_to_padded as character_to_padded,
    make_corpus_operations as make_character_operations
)

# word-level specific imports
from src.word_level.utils import (
    input_to_padded as word_to_padded,
    make_corpus_operations as make_word_operations
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config(model_type: str, corpus: str):
    if model_type == 'character':
        operations = make_character_operations(corpus)
        input_to_padded = character_to_padded
    elif model_type == 'word':
        operations = make_word_operations(corpus)
        input_to_padded = word_to_padded
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'input_to_padded': input_to_padded,
        'input_to_index': operations['input_to_index'],
        'index_to_token': operations['index_to_token'],
        'token_to_index': operations['token_to_index']
    }

def apply_temperature(logits_probs, temperature: float = 1.0):
    logits_probs = logits_probs / temperature
    return torch.softmax(logits_probs, dim=-1)

def apply_repetition_penalty(logits_probs, generated_indices, penalty=1.0):
    for index in generated_indices:
        logits_probs[index] /= penalty
    return logits_probs

def top_k_sampling(logits_probs, k: int = 10):
    top_k_probs, top_k_indices = torch.topk(logits_probs, k)
    top_k_probs = top_k_probs / top_k_probs.sum()

    token_index = torch.multinomial(top_k_probs, 1).item()

    return top_k_indices[token_index].item()

def top_p_sampling(logits_probs, p: float = 0.9):
    sorted_probs, sorted_indices = torch.sort(logits_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    cutoff_index = torch.where(cumulative_probs > p)[0][0].item()
    
    top_p_probs = sorted_probs[:cutoff_index + 1]
    top_p_probs = top_p_probs / top_p_probs.sum()
    top_p_indices = sorted_indices[:cutoff_index + 1]

    token_index = torch.multinomial(top_p_probs, 1).item()

    return top_p_indices[token_index].item()

def make_evaluator(
    model: nn.Module,
    input_to_padded,
    input_to_index,
    index_to_token: dict,
    token_to_index: dict
):
    def eval(input: str, sequence_size, output_size: int = 255) -> str:
        output = input.lower().split()
        padded_input = input_to_padded(input, sequence_size)

        indices = input_to_index(padded_input).to(device)

        model.eval()

        token = None
        i = 0
        while i < output_size and token != END_OF_THOUGHT_TOKEN:
            with torch.no_grad():
                logits = model(indices)

                logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                logits_probs = apply_temperature(logits_probs, temperature=1.0)
                logits_probs = apply_repetition_penalty(logits_probs, [token_to_index.get(char, token_to_index[UNKNOWN_TOKEN]) for char in output], penalty=1.0)

                token_index = top_k_sampling(logits_probs, k=10)
                token = index_to_token[token_index]

            if token != UNKNOWN_TOKEN:
                output.append(token)

            next_index = input_to_index(token).to(device)

            indices = torch.cat((indices[:, -sequence_size+1:], next_index), dim=1)

            i += 1

        return ' '.join(output)
    
    return eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text based on a arbitrary corpus.")
    
    parser.add_argument('--model_name', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--model_type', type=str, required=True, choices=['character', 'word'],
                        help='Specify whether to train on a character-level or word-level model.')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--prompt_text', type=str, required=True,
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--output_size', type=int, required=True,
                        help='The size of the generated text output (e.g., "100")')

    args = parser.parse_args()

    eval = make_evaluator(
        model=load_model_from_archive(device, args.model_name, args.model_type),
        **get_config(args.model_type, corpus=fetch_and_load_corpus(args.url))
    )

    print(eval(args.prompt_text, args.sequence_size, args.output_size))