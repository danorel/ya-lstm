import torch

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str):
    chars = set(corpus)

    vocab = sorted(chars) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    char_to_index = {c: i for i, c in enumerate(vocab)}
    index_to_char = {i: c for i, c in enumerate(vocab)}

    def input_to_index(prompt: str) -> torch.Tensor:
        indices = [char_to_index.get(char, char_to_index[UNKNOWN_TOKEN]) for char in prompt]
        return torch.tensor(indices).long().unsqueeze(0)

    return {
        'input_to_index': input_to_index,
        'index_to_token': index_to_char,
        'token_to_index': char_to_index,
        'vocab': vocab,
        'vocab_size': vocab_size
    }

def input_to_padded(prompt: str, sequence_size: int) -> list[str]:
    if len(prompt) < sequence_size:
        padding_size = sequence_size - len(prompt)
        padded_prompt = [PAD_TOKEN] * padding_size + list(prompt)
    else:
        padded_prompt = list(prompt[-sequence_size:])
    return padded_prompt