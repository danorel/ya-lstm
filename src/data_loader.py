import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

class CorpusDataset(Dataset):
    def __init__(self, corpus, char_to_index, sequence_size):
        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([char_to_index[char] for char in corpus]),
            window_shape=sequence_size
        )
        self.targets = np.ascontiguousarray(sequences_windows[:, 1:])
        self.sequences = np.ascontiguousarray(sequences_windows[:, :-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.targets[idx])

def create_dataloader(corpus: str, char_to_index, sequence_size, batch_size, num_workers: int = 0):
    dataset = CorpusDataset(corpus, char_to_index, sequence_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader