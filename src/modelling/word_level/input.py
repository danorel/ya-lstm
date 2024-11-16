from src.constants.corpus import PAD_TOKEN


def input_to_padded(text: str, sequence_size: int) -> list[str]:
    words = text.split()
    if len(words) < sequence_size:
        padding_size = sequence_size - len(words)
        padded_text = [PAD_TOKEN] * padding_size + words
    else:
        padded_text = words[-sequence_size:]
    return padded_text
