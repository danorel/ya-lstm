import argparse
import math

from src.constants.modelling import ArchitectureName, ModellingName
from src.corpus.loader import fetch_and_load_corpus
from src.modelling.character_level.corpus import get_utils as get_character_level_corpus_utils
from src.modelling.common.model_arguments import Metadata
from src.modelling.common.model_trainer import Architecture, CorpusUtils, Hyperparameters, Plugins, Setup, make_trainer
from src.modelling.word_level.corpus import get_utils as make_word_level_corpus_utils


def get_corpus_utils(modelling_name: str, corpus: str) -> CorpusUtils:
    if modelling_name == ModellingName.CHARACTER_LEVEL.value:
        return get_character_level_corpus_utils(corpus)
    if modelling_name == ModellingName.WORD_LEVEL.value:
        return make_word_level_corpus_utils(corpus)
    raise ValueError(f"Unknown model type: {modelling_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM-like model on a specified text corpus.")

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
    parser.add_argument("--use_profiler", action="store_true", help="Enable profiling with torch.utils.bottleneck")
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable tensorboard metrics collection")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for training.")
    parser.add_argument("--cells_size", type=int, default=3, help="The number of LSTM layers.")
    parser.add_argument("--embedding_size", type=int, default=512, help="The size of embedding layers.")
    parser.add_argument("--hidden_size", type=int, default=1024, help="The size of hidden/context layers.")
    parser.add_argument("--sequence_size", type=int, default=64, help="The size of each input sequence.")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of samples in each batch.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate parameter of LSTM optimizer.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma parameter of LSTM optimizer's scheduler serving for training stabilization.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay parameter of LSTM optimizer serving for weights normalization to avoid overfitting.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping serving for weights update normalization to avoid gradient explosion and overflow.",
    )
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument(
        "--max_steps", type=float, default=+math.inf, help="Number of training steps within the epoch."
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Accumulations steps per LSTM optimizer to make backward propagation pass (Adam is a default setting).",
    )
    parser.add_argument(
        "--patience_steps",
        type=int,
        default=100,
        help="Patience steps to verify whether model has not stopped learning.",
    )

    args = parser.parse_args()

    train_fn = make_trainer(
        Metadata(
            architecture_name=args.architecture_name,
            modelling_name=args.modelling_name,
        ),
        Plugins(use_tensorboard=args.use_tensorboard),
        Setup(
            max_epochs=(1 if args.use_profiler else args.max_epochs),
            max_steps=(100 if args.use_profiler else args.max_steps),
            accumulation_steps=args.accumulation_steps,
            patience_steps=args.patience_steps,
        ),
        get_corpus_utils(args.modelling_name, corpus=fetch_and_load_corpus(args.url)),
    )

    try:
        train_fn(
            Architecture(
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                cells_size=args.cells_size,
                dropout=args.dropout,
            ),
            Hyperparameters(
                sequence_size=args.sequence_size,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
            ),
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as exception:
        print(f"Training failed with error: {exception}")
