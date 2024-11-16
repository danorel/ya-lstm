import argparse
import json
from pathlib import Path

import optuna

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
    raise ValueError(f"Unknown modelling name: {modelling_name}")


# pylint: disable=redefined-outer-name
def objective(train_fn):
    def experiment(trial):
        accuracy = train_fn(
            Architecture(
                dropout=trial.suggest_float("dropout", 0.1, 0.5),
                cells_size=trial.suggest_int("cells_size", 1, 3),
                embedding_size=trial.suggest_int("embedding_size", 128, 512),
                hidden_size=trial.suggest_int("hidden_size", 128, 512),
            ),
            Hyperparameters(
                sequence_size=trial.suggest_int("sequence_size", 16, 64),
                batch_size=trial.suggest_int("batch_size", 128, 1024),
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                gamma=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
                grad_clip_norm=trial.suggest_float("grad_clip_norm", 1.0, 5.0),
            ),
        )
        return accuracy

    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")

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
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable tensorboard metrics collection")
    parser.add_argument(
        "--trials", type=int, required=True, help="Amount of hyperparameter search iterations (e.g., 1, 4, 8, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="models/lstm/best_trial.json",
        help='Output file to store the best trial parameters (e.g., "models/lstm/best_trial.json")',
    )

    args = parser.parse_args()

    train_fn = make_trainer(
        Metadata(
            architecture_name=args.architecture_name,
            modelling_name=args.modelling_name,
        ),
        Plugins(use_tensorboard=args.use_tensorboard),
        Setup(
            max_epochs=1,
            max_steps=50,
            accumulation_steps=3,
            patience_steps=100,
        ),
        get_corpus_utils(args.modelling_name, fetch_and_load_corpus(args.url)),
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective(train_fn), n_trials=args.trials)

    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=4)

    print(f"Best trial parameters saved to {args.output}")
