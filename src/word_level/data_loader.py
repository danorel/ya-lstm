import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

from src.core.constants import UNKNOWN_TOKEN

class WordDataset(Dataset):
    def __init__(self, corpus: str, index_from_word: dict, sequence_size: int):
        words = corpus.split()

        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([index_from_word.get(word, index_from_word[UNKNOWN_TOKEN]) for word in words]),
            window_shape=sequence_size
        )

        self.targets = np.ascontiguousarray(sequences_windows[:, 1:])
        self.sequences = np.ascontiguousarray(sequences_windows[:, :-1])

        assert self.sequences.shape[1] == sequence_size - 1, \
            f"Sequences should have length {sequence_size - 1}, got {self.sequences.shape[1]}"
        assert self.targets.shape[1] == sequence_size - 1, \
            f"Targets should have length {sequence_size - 1}, got {self.targets.shape[1]}"

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.targets[idx])

def create_dataloader(corpus: str, index_from_word: dict, sequence_size: int, batch_size: int, num_workers: int = 0):
    dataset = WordDataset(corpus, index_from_word, sequence_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader