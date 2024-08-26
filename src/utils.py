import torch
import torch.nn.functional as F

def embedding_from_indices(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    embedding = F.one_hot(indices, num_classes=vocab_size)
    return embedding.float()


def embedding_from_prompt(prompt: str, char_to_index: dict, vocab_size: int) -> torch.Tensor:
    indices = torch.tensor([char_to_index[char] for char in prompt], dtype=torch.long)
    embedding = embedding_from_indices(indices, vocab_size)
    return embedding