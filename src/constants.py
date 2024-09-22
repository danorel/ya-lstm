# model configurations
CORPUS_DIR = "corpus"
MODEL_ARCHIVE_DIR = "model_archive"
MODEL_REPOSITORY_DIR = "model_repository"

# corpus configurations
PAD_CHAR = '<PAD>'
UNKNOWN_CHAR = '<UNK>'
VOCAB = sorted(set(['\n', ' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '}', '~'])) + [UNKNOWN_CHAR, PAD_CHAR]
VOCAB_SIZE = len(VOCAB)
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(VOCAB)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}