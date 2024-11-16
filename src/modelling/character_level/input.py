from src.constants.corpus import PAD_TOKEN


def input_to_padded(text: str, sequence_size: int) -> list[str]:
    if len(text) < sequence_size:
        padding_size = sequence_size - len(text)
        padded_text = [PAD_TOKEN] * padding_size + list(text)
    else:
        padded_text = list(text[-sequence_size:])
    return padded_text
