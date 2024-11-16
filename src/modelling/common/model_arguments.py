from collections.abc import Callable
from dataclasses import dataclass

from src.constants.modelling import ArchitectureName, ModellingName


@dataclass
class Metadata:
    architecture_name: ArchitectureName
    modelling_name: ModellingName


@dataclass
class CorpusUtils:
    vocab: list[str]
    vocab_size: int
    input_to_index: Callable
    index_to_token: dict
    token_to_index: dict
    create_dataloader: Callable
