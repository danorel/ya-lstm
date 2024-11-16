from torch.utils.data import DataLoader

from src.modelling.word_level.dataset import WordDataset


def setup_dataloader(corpus: str, word_to_index: dict):
    def create_dataloader(sequence_size: int, batch_size: int):
        dataset = WordDataset(corpus, word_to_index, sequence_size)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return dataloader

    return create_dataloader
