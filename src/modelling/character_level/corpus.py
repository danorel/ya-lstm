import torch

from src.constants.corpus import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN
from src.modelling.character_level.data_loader import setup_dataloader
from src.modelling.common.model_trainer import CorpusUtils


def get_utils(corpus: str) -> CorpusUtils:
    chars = set(corpus)

    vocab = sorted(chars) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    char_to_index = {c: i for i, c in enumerate(vocab)}
    index_to_char = dict(enumerate(vocab))

    def input_to_index(prompt: str) -> torch.Tensor:
        indices = [char_to_index.get(char, char_to_index[UNKNOWN_TOKEN]) for char in prompt]
        return torch.tensor(indices).long().unsqueeze(0)

    return CorpusUtils(
        create_dataloader=setup_dataloader(corpus, char_to_index),
        input_to_index=input_to_index,
        index_to_token=index_to_char,
        token_to_index=char_to_index,
        vocab=vocab,
        vocab_size=vocab_size,
    )
