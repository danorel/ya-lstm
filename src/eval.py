import argparse
import torch
import torch.nn as nn

from src.core.constants import UNKNOWN_TOKEN, END_OF_THOUGHT_TOKEN
from src.core.corpus_loader import fetch_and_load_corpus
from src.core.model import load_model_from_archive

# character-level specific imports
from src.character_level.utils import (
    create_prompt as create_character_prompt,
    make_corpus_operations as make_character_operations
)

# word-level specific imports
from src.word_level.utils import (
    create_prompt as create_word_prompt,
    make_corpus_operations as make_word_operations
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config(model_type: str, corpus: str):
    if model_type == 'character':
        operations = make_character_operations(corpus)
        create_prompt = create_character_prompt
    elif model_type == 'word':
        operations = make_word_operations(corpus)
        create_prompt = create_word_prompt
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'create_prompt': create_prompt,
        'create_input_from_prompt': operations['create_input_from_prompt'],
        'index_to_token': operations['index_to_token'],
    }

def make_evaluator(
    model: nn.Module,
    create_prompt,
    create_input_from_prompt,
    index_to_token: dict,
):
    def eval(prompt: str, sequence_size, output_size: int = 255) -> str:
        output_prompt = prompt.split()
        padded_prompt = create_prompt(prompt, sequence_size)

        sequence = create_input_from_prompt(padded_prompt)
        sequence = sequence.unsqueeze(0).to(device)

        model.eval()

        token = None
        i = 0
        while i < (output_size - len(prompt)) and token != END_OF_THOUGHT_TOKEN:
            with torch.no_grad():
                logits = model(sequence)
                logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()

            token_idx = torch.multinomial(logits_probs, 1).item()
            token = index_to_token[token_idx]

            if token != UNKNOWN_TOKEN:
                output_prompt.append(token)

            symbol = create_input_from_prompt(token)
            symbol = symbol.unsqueeze(0).to(device)

            sequence = torch.cat((sequence[:, -sequence_size+1:], symbol), dim=1)

            i += 1

        return ' '.join(output_prompt)
    
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