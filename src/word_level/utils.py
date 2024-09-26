import torch

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str) -> dict:
    words = set(corpus.split())

    vocab = sorted(words) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    def input_to_index(prompt: list[str]) -> torch.Tensor:
        indices = [word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in prompt]
        return torch.tensor(indices).long().unsqueeze(0)
    
    return {
        'input_to_index': input_to_index,
        'index_to_token': index_to_word,
        'token_to_index': word_to_index,
        'vocab': vocab,
        'vocab_size': vocab_size
    }

def input_to_padded(prompt: str, sequence_size: int) -> list[str]:
    words = prompt.split()
    if len(words) < sequence_size:
        padding_size = sequence_size - len(words)
        padded_prompt = [PAD_TOKEN] * padding_size + words
    else:
        padded_prompt = words[-sequence_size:]
    return padded_prompt