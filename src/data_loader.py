import numpy as np
import torch
import typing as t

from src.utils import one_hot_encoding

def dataloader(corpus: str, char_to_index: dict, vocab_size: int, sequence_size: int, batch_size: int, start_index: int = 0, end_index: t.Optional[int] = None):
    if end_index is None:
        end_index = len(corpus) - 1

    corpus_indices = np.array([char_to_index[char] for char in corpus])
    
    sequences = [corpus_indices[i:i+sequence_size] for i in range(start_index, end_index - sequence_size)]
    targets = [corpus_indices[i+1:i+1+sequence_size] for i in range(start_index, end_index - sequence_size)]

    total_sequences = len(sequences)
    for i in range(0, total_sequences, batch_size):
        batch_sequences = sequences[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        if len(batch_sequences) == batch_size:
            yield one_hot_encoding(torch.tensor(batch_sequences, dtype=torch.long), vocab_size), torch.tensor(batch_targets, dtype=torch.long)