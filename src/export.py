import argparse
import torch

from src.constants import VOCAB_SIZE
from src.model import load_model_from_archive, load_model_to_repository

def export(name: str, sequence_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_archive(device, name)

    example_embedding = torch.randn(1, sequence_size, VOCAB_SIZE).to(device)

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
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')

    args = parser.parse_args()

    export(args.name, args.sequence_size)