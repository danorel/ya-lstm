import argparse
import torch
import torch.nn as nn

from src.core.model import load_model_from_archive
from src.core.corpus_loader import fetch_and_load_corpus

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
    
    (
        create_embedding_from_indices,
        create_embedding_from_prompt,
        token_to_index,
        index_to_token,
        vocab,
        vocab_size
    ) = operations

    return {
        'create_prompt': create_prompt,
        'create_embedding_from_prompt': create_embedding_from_prompt,
        'create_embedding_from_indices': create_embedding_from_indices,
        'index_to_token': index_to_token,
    }

def make_evaluator(
    model: nn.Module,
    create_prompt,
    create_embedding_from_prompt,
    create_embedding_from_indices,
    index_to_token: dict,
):
    def eval(prompt: str, sequence_size, output_size: int = 255) -> str:
        padded_prompt = create_prompt(prompt, sequence_size)

        sequence_embedding = create_embedding_from_prompt(padded_prompt)
        sequence_embedding = sequence_embedding.unsqueeze(0).to(device)

        model.eval()

        token = None
        i = 0
        while i < (output_size - len(prompt)) and token != '\n':
            with torch.no_grad():
                logits = model(sequence_embedding)
                logits_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()

            token_idx = torch.multinomial(logits_probs, 1).item()
            token = index_to_token[token_idx]
            padded_prompt.append(token)

            char_embedding = create_embedding_from_indices(torch.tensor([token_idx]))
            char_embedding = char_embedding.unsqueeze(0).to(device)
            sequence_embedding = torch.cat((sequence_embedding[:, -sequence_size+1:, :], char_embedding), dim=1)

            i += 1

        return padded_prompt
    
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
        model=load_model_from_archive(device, args.model_name, args.model_type)
        **get_config(args.model_type, corpus=fetch_and_load_corpus(args.url))
    )

    print(eval(args.prompt_text, args.sequence_size, args.output_size))