import torch
import torch.nn.functional as F

from src.core.constants import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN


def make_corpus_operations(corpus: str):
    words = set(corpus.split())

    vocab = sorted(words) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    def create_embedding_from_indices(indices: torch.Tensor) -> torch.Tensor:
        embedding = F.one_hot(indices, num_classes=vocab_size).float()
        embedding[indices == word_to_index[PAD_TOKEN]] = 0.0
        return embedding
    
    def create_embedding_from_prompt(prompt) -> torch.Tensor:
        sequence = torch.tensor([word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in prompt], dtype=torch.long)
        embedding = create_embedding_from_indices(sequence)
        return embedding
    
    return (
        create_embedding_from_indices,
        create_embedding_from_prompt,
        word_to_index,
        index_to_word,
        vocab,
        vocab_size
    )

def create_prompt(prompt: str, sequence_size: int) -> list[str]:
    words = prompt.split()
    if len(words) < sequence_size:
        padding_size = sequence_size - len(words)
        padded_prompt = [PAD_TOKEN] * padding_size + words
    else:
        padded_prompt = words[-sequence_size:]
    return padded_prompt