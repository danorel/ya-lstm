import torch
import torch.nn.functional as F

from src.constants import CHAR_TO_INDEX, INDEX_TO_CHAR, VOCAB_SIZE, PAD_CHAR, UNKNOWN_CHAR


def index_from_char(char: str) -> int:
    return CHAR_TO_INDEX.get(char, CHAR_TO_INDEX[UNKNOWN_CHAR])

def char_from_index(index: int) -> str:
    return INDEX_TO_CHAR[index]

def embedding_from_indices(indices: torch.Tensor) -> torch.Tensor:
    embedding = F.one_hot(indices, num_classes=VOCAB_SIZE).float()
    embedding[indices == index_from_char(PAD_CHAR)] = 0.0
    return embedding

def embedding_from_prompt(prompt) -> torch.Tensor:
    indices = torch.tensor([index_from_char(char) for char in prompt], dtype=torch.long)
    embedding = embedding_from_indices(indices)
    return embedding