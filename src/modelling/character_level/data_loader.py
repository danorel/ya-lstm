from torch.utils.data import DataLoader

from src.modelling.character_level.dataset import CharacterDataset


def setup_dataloader(corpus: str, char_to_index: dict):
    def create_dataloader(sequence_size: int, batch_size: int):
        dataset = CharacterDataset(corpus, char_to_index, sequence_size)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return dataloader

    return create_dataloader
