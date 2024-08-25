import numpy as np
import torch
import typing as t

from src.utils import one_hot_encoding

def dataloader(device: torch.device, corpus: str, char_to_index: dict, vocab_size: int, sequence_size: int, batch_size: int, start_index: int = 0, end_index: t.Optional[int] = None):
    if end_index is None:
        end_index = len(corpus) - 1

    corpus_indices = np.array([char_to_index[char] for char in corpus])

    sequences = np.lib.stride_tricks.sliding_window_view(corpus_indices[start_index:end_index], window_shape=sequence_size)
    targets = sequences[:, 1:]
    sequences = sequences[:, :-1]

    sequences_tensor = torch.tensor(sequences, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    num_sequences = end_index - start_index - sequence_size

    for i in range(0, num_sequences, batch_size):
        batch_sequences = sequences_tensor[i:i+batch_size]
        batch_targets = targets_tensor[i:i+batch_size]
        
        if batch_sequences.size(0) == batch_size:
            batch_embedding = one_hot_encoding(
                indices=batch_sequences,
                vocab_size=vocab_size
            ).to(device)
            yield batch_embedding, batch_targets