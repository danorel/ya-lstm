import torch
import torch.nn.functional as F

def one_hot_encoding(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    one_hot_encoded = F.one_hot(indices, num_classes=vocab_size)
    return one_hot_encoded.float()