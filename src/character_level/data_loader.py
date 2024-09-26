import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

from src.core.constants import UNKNOWN_TOKEN

class CharacterDataset(Dataset):
    def __init__(self, device, corpus: str, char_to_index: dict, sequence_size: int):
        self.device = device

        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([char_to_index.get(char, char_to_index[UNKNOWN_TOKEN]) for char in corpus]),
            window_shape=sequence_size
        )

        self.targets = np.ascontiguousarray(sequences_windows[:, 1:])
        self.indices = np.ascontiguousarray(sequences_windows[:, :-1])

        assert self.indices.shape[1] == sequence_size - 1, \
            f"Indices should have length {sequence_size - 1}, got {self.indices.shape[1]}"
        assert self.targets.shape[1] == sequence_size - 1, \
            f"Targets should have length {sequence_size - 1}, got {self.targets.shape[1]}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.indices[idx]).long().to(self.device),
            torch.from_numpy(self.targets[idx]).long().to(self.device)
        )

def create_dataloader(device, corpus: str, char_to_index: dict, sequence_size: int, batch_size: int, num_workers: int = 0):
    dataset = CharacterDataset(device, corpus, char_to_index, sequence_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader