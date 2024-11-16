import argparse

from src.constants.modelling import ArchitectureName, ModellingName
from src.corpus.loader import fetch_and_load_corpus
from src.modelling.character_level.corpus import get_utils as get_character_level_corpus_utils
from src.modelling.common.model_arguments import Metadata
from src.modelling.common.model_exporter import make_exporter
from src.modelling.word_level.corpus import get_utils as make_word_level_corpus_utils


def get_corpus_utils(modelling_name: str, corpus: str):
    if modelling_name == ModellingName.CHARACTER_LEVEL.value:
        return get_character_level_corpus_utils(corpus)
    if modelling_name == ModellingName.WORD_LEVEL.value:
        return make_word_level_corpus_utils(corpus)
    raise ValueError(f"Unknown model type: {modelling_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export an RNN-based model trained on the specified text corpus.")

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
    parser.add_argument("--sequence_size", type=int, default=16, help="The size of each input sequence")

    args = parser.parse_args()

    export_fn = make_exporter(
        Metadata(
            architecture_name=args.architecture_name,
            modelling_name=args.modelling_name,
        ),
        get_corpus_utils(args.modelling_name, corpus=fetch_and_load_corpus(args.url)),
    )

    try:
        export_fn(args.sequence_size)
    except KeyboardInterrupt:
        print("Exporting interrupted by user.")
    except Exception as exception:
        print(f"Exporting failed with error: {exception}")
