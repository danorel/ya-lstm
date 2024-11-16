import argparse

from src.constants.modelling import ArchitectureName, ModellingName
from src.corpus.loader import fetch_and_load_corpus
from src.modelling.character_level.corpus import get_utils as get_character_level_corpus_utils
from src.modelling.character_level.input import input_to_padded as character_to_padded
from src.modelling.common.model_arguments import CorpusUtils, Metadata
from src.modelling.common.model_prompter import make_prompter
from src.modelling.word_level.corpus import get_utils as make_word_level_training_utils
from src.modelling.word_level.input import input_to_padded as word_to_padded


def get_corpus_utils(modelling_name: str, corpus: str) -> CorpusUtils:
    if modelling_name == ModellingName.CHARACTER_LEVEL.value:
        return get_character_level_corpus_utils(corpus)
    if modelling_name == ModellingName.WORD_LEVEL.value:
        return make_word_level_training_utils(corpus)
    raise ValueError(f"Unknown model type: {modelling_name}")


def get_token_to_padded(modelling_name: str):
    if modelling_name == ModellingName.CHARACTER_LEVEL.value:
        return character_to_padded
    if modelling_name == ModellingName.WORD_LEVEL.value:
        return word_to_padded
    raise ValueError(f"Unknown model type: {modelling_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text based on a arbitrary corpus.")

    parser.add_argument(
        "--architecture_name",
        type=str,
        required=True,
        choices=[ArchitectureName.LSTM.value, ArchitectureName.GRU.value],
        help='Model to use as a basis for text generation (e.g., "lstm")',
    )
    parser.add_argument(
        "--modelling_name",
        type=str,
        required=True,
        choices=[ModellingName.CHARACTER_LEVEL.value, ModellingName.WORD_LEVEL.value],
        help="Specify whether to train on a character-level or word-level model.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt",
        help="URL to fetch the text corpus",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        required=True,
        help='Text to use as a basis for text generation (e.g., "Forecasting for you")',
    )
    parser.add_argument("--sequence_size", type=int, default=16, help="The size of each input sequence")
    parser.add_argument(
        "--output_size", type=int, required=True, help='The size of the generated text output (e.g., "100")'
    )

    args = parser.parse_args()

    prompt_fn = make_prompter(
        Metadata(architecture_name=args.architecture_name, modelling_name=args.modelling_name),
        get_token_to_padded(args.modelling_name),
        get_corpus_utils(args.modelling_name, corpus=fetch_and_load_corpus(args.url)),
    )

    try:
        print(prompt_fn(args.input_text, args.sequence_size, args.output_size))
    except KeyboardInterrupt:
        print("Prompting interrupted by user.")
    except Exception as exception:
        print(f"Prompting failed with error: {exception}")
