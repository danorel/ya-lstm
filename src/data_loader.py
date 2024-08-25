import multiprocessing
import numpy as np

from torch.utils.data import DataLoader, Dataset

class CorpusDataset(Dataset):
    def __init__(self, corpus, char_to_index, sequence_size):
        sequences_windows = np.lib.stride_tricks.sliding_window_view(
            x=np.array([char_to_index[char] for char in corpus]),
            window_shape=sequence_size
        )
        self.targets = sequences_windows[:, 1:]
        self.sequences = sequences_windows[:, :-1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_dataloader(corpus: str, char_to_index, sequence_size, batch_size, num_workers: int = int(multiprocessing.cpu_count() // 2)):
    dataset = CorpusDataset(corpus, char_to_index, sequence_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader