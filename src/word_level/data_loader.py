import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

from src.core.constants import UNKNOWN_TOKEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordDataset(Dataset):
    def __init__(self, corpus: str, word_to_index: dict, sequence_size: int):
        words = corpus.split()

        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in words]),
            window_shape=sequence_size
        )

        self.targets = np.ascontiguousarray(sequences_windows[:, 1:])
        self.indices = np.ascontiguousarray(sequences_windows[:, :-1])

        assert self.indices.shape[1] == sequence_size - 1, \
            f"Sequences should have length {sequence_size - 1}, got {self.indices.shape[1]}"
        assert self.targets.shape[1] == sequence_size - 1, \
            f"Targets should have length {sequence_size - 1}, got {self.targets.shape[1]}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.indices[idx]).long().to(device),
            torch.from_numpy(self.targets[idx]).long().to(device)
        )

def make_dataloader(corpus: str, word_to_index: dict):
    def use_dataloader(sequence_size: int, batch_size: int):
        dataset = WordDataset(corpus, word_to_index, sequence_size)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return dataloader
    
    return use_dataloader