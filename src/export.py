import argparse
import torch

from src.corpus_loader import fetch_and_load_corpus
from src.model import load_model_from_archive, load_model_to_repository
from src.utils import embedding_from_prompt

def export(name: str, url: str, prompt_text: str, sequence_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = fetch_and_load_corpus(url)

    vocab = sorted(set(corpus))
    vocab_size = len(vocab)
    char_to_index = {char: idx for idx, char in enumerate(vocab)}

    model = load_model_from_archive(device, name)

    example_embedding = embedding_from_prompt(prompt_text[-sequence_size:].lower(), char_to_index, vocab_size)
    example_embedding = example_embedding.unsqueeze(0).to(device)

    load_model_to_repository(
        model,
        example_embedding,
        name,
        version=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export an RNN-based model trained on the specified text corpus.")
    
    parser.add_argument('--name', type=str, required=True, choices=['lstm', 'gru'],
                    help='Model to use as a basis for text generation (e.g., "bidirectional")')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--prompt_text', type=str, default='Hello my dear darling and princess, I am fond of you, and you know it very ',
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you ")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')

    args = parser.parse_args()

    export(args.name, args.url, args.prompt_text, args.sequence_size)