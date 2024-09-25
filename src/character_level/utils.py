import torch
import torch.nn.functional as F

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str):
    chars = set(corpus)

    vocab = sorted(chars) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    char_to_index = {c: i for i, c in enumerate(vocab)}
    index_to_char = {i: c for i, c in enumerate(vocab)}

    def create_embedding_from_indices(indices: torch.Tensor) -> torch.Tensor:
        embedding = F.one_hot(indices, num_classes=vocab_size).float()
        embedding[indices == char_to_index[PAD_TOKEN]] = 0.0
        return embedding

    def create_embedding_from_prompt(prompt: str) -> torch.Tensor:
        indices = torch.tensor([char_to_index.get(char, char_to_index[UNKNOWN_TOKEN]) for char in prompt], dtype=torch.long)
        embedding = create_embedding_from_indices(indices)
        return embedding

    return (
        create_embedding_from_indices,
        create_embedding_from_prompt,
        char_to_index,
        index_to_char,
        vocab,
        vocab_size
    )

def create_prompt(prompt: str, sequence_size: int) -> list[str]:
    if len(prompt) < sequence_size:
        padding_size = sequence_size - len(prompt)
        padded_prompt = [PAD_TOKEN] * padding_size + list(prompt)
    else:
        padded_prompt = list(prompt[-sequence_size:])
    return padded_prompt