import numpy as np
import torch
from torch.utils.data import Dataset

from src.constants.corpus import UNKNOWN_TOKEN
from src.constants.device import device


class WordDataset(Dataset):
    def __init__(self, corpus: str, word_to_index: dict, sequence_size: int):
        words = corpus.split()

        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in words]),
            window_shape=sequence_size,
        )

        self.targets = np.ascontiguousarray(sequences_windows[:, 1:])
        self.indices = np.ascontiguousarray(sequences_windows[:, :-1])

        assert (
            self.indices.shape[1] == sequence_size - 1
        ), f"Sequences should have length {sequence_size - 1}, got {self.indices.shape[1]}"
        assert (
            self.targets.shape[1] == sequence_size - 1
        ), f"Targets should have length {sequence_size - 1}, got {self.targets.shape[1]}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.indices[idx]).long().to(device),
            torch.from_numpy(self.targets[idx]).long().to(device),
        )
