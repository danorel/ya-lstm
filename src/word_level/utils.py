import torch

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str) -> dict:
    words = set(corpus.split())

    vocab = sorted(words) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    def create_input_from_prompt(prompt: list[str]) -> torch.Tensor:
        indices = torch.tensor([word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in prompt]).long()
        return indices
    
    return {
        'create_input_from_prompt': create_input_from_prompt,
        'token_to_index': word_to_index,
        'index_to_token': index_to_word,
        'vocab': vocab,
        'vocab_size': vocab_size
    }

def create_prompt(prompt: str, sequence_size: int) -> list[str]:
    words = prompt.split()
    if len(words) < sequence_size:
        padding_size = sequence_size - len(words)
        padded_prompt = [PAD_TOKEN] * padding_size + words
    else:
        padded_prompt = words[-sequence_size:]
    return padded_prompt