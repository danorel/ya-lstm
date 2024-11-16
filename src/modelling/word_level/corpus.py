import torch

from src.constants.corpus import END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN
from src.modelling.common.model_trainer import CorpusUtils
from src.modelling.word_level.data_loader import setup_dataloader


def get_utils(corpus: str) -> CorpusUtils:
    words = set(corpus.split())

    vocab = sorted(words) + [END_OF_THOUGHT_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN]
    vocab_size = len(vocab)

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = dict(enumerate(vocab))

    def input_to_index(prompt: list[str]) -> torch.Tensor:
        indices = [word_to_index.get(word, word_to_index[UNKNOWN_TOKEN]) for word in prompt]
        return torch.tensor(indices).long().unsqueeze(0)

    return CorpusUtils(
        create_dataloader=setup_dataloader(corpus, word_to_index),
        input_to_index=input_to_index,
        index_to_token=index_to_word,
        token_to_index=word_to_index,
        vocab=vocab,
        vocab_size=vocab_size,
    )
