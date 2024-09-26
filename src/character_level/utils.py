import torch
import torch.nn.functional as F

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str):
    chars = set(corpus)

    vocab = sorted(chars) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    char_to_index = {c: i for i, c in enumerate(vocab)}
    index_to_char = {i: c for i, c in enumerate(vocab)}

    def create_input_from_prompt(prompt: str) -> torch.Tensor:
        indices = torch.tensor([char_to_index.get(char, char_to_index[UNKNOWN_TOKEN]) for char in prompt]).long()
        return indices

    return {
        'create_input_from_prompt': create_input_from_prompt,
        'token_to_index': char_to_index,
        'index_to_token': index_to_char,
        'vocab': vocab,
        'vocab_size': vocab_size
    }

def create_prompt(prompt: str, sequence_size: int) -> list[str]:
    if len(prompt) < sequence_size:
        padding_size = sequence_size - len(prompt)
        padded_prompt = [PAD_TOKEN] * padding_size + list(prompt)
    else:
        padded_prompt = list(prompt[-sequence_size:])
    return padded_prompt