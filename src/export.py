import argparse
import torch

from src.core.corpus_loader import fetch_and_load_corpus
from src.core.model import load_model_from_archive, load_model_to_repository

from src.character_level.utils import make_corpus_operations as make_character_utils
from src.word_level.utils import make_corpus_operations as make_word_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_exporting_utils(model_type: str, corpus: str):
    if model_type == 'character':
        utils = make_character_utils(corpus)
    elif model_type == 'word':
        utils = make_word_utils(corpus)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return utils

def make_exporter(
    model_name: str,
    model_type: str,
    vocab_size: int,
    **kwargs
):
    model = load_model_from_archive(model_name, model_type),

    def export(sequence_size: int):
        load_model_to_repository(
            model,
            example=torch.randn(1, sequence_size, vocab_size).to(device),
            path=f"{model_name}-{model_type}",
            version="1",
        )
    
    return export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export an RNN-based model trained on the specified text corpus.")
    
    parser.add_argument('--model_name', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--model_type', type=str, required=True, choices=['character', 'word'],
                    help='Specify whether to train on a character-level or word-level model.')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                    help='URL to fetch the text corpus')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')

    args = parser.parse_args()

    export = make_exporter(
        args.model_name,
        args.model_type,
        **get_exporting_utils(
            args.model_type,
            corpus=fetch_and_load_corpus(args.url)
        )
    )
    export(args.sequence_size)