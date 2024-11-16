from enum import Enum


class ArchitectureName(Enum):
    LSTM = "lstm"
    GRU = "gru"


class ModellingName(Enum):
    CHARACTER_LEVEL = "character_level"
    WORD_LEVEL = "word_level"
