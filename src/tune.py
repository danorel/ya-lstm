import argparse
import json
import optuna
import torch

from pathlib import Path

from src.train import (
    ModelArchitecture,
    ModelHyperparameters,
    ModelTrainingConfig,
    get_training_utils,
    make_trainer
)
from src.core.corpus_loader import fetch_and_load_corpus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(train):
    def experiment(trial):
        accuracy = train(
            hidden_architecture=ModelArchitecture(
                dropout=trial.suggest_float('dropout', 0.1, 0.5),
                cells_size=trial.suggest_int('cells_size', 1, 3),
                embedding_size=trial.suggest_int('embedding_size', 128, 512),
                hidden_size=trial.suggest_int('hidden_size', 128, 512),
            ),
            hyperparameters=ModelHyperparameters(
                sequence_size=trial.suggest_int('sequence_size', 16, 64),
                batch_size=trial.suggest_int('batch_size', 128, 1024),
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                gamma=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                weight_decay=trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
                grad_clip_norm=trial.suggest_float('grad_clip_norm', 1.0, 5.0)
            )
        )
        return accuracy
    
    return experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--model_name', type=str, required=True, choices=['lstm', 'gru'],
                        help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--model_type', type=str, required=True, choices=['character', 'word'],
                        help='Specify whether to train on a character-level or word-level model.')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help="Enable tensorboard metrics collection")
    parser.add_argument('--trials', type=int, required=True,
                    help='Amount of hyperparameter search iterations (e.g., 1, 4, 8, etc.)')
    parser.add_argument('--output', type=str, required=False, default='models/lstm/best_trial.json',
                        help='Output file to store the best trial parameters (e.g., "models/lstm/best_trial.json")')

    args = parser.parse_args()

    print(f"Tuning config:\n\ttensorboard = {args.use_tensorboard}\n")

    train = make_trainer(
        args.model_name,
        args.model_type,
        use_tensorboard=args.use_tensorboard,
        training_config=ModelTrainingConfig(
            max_epochs=1,
            max_steps=51,
            accumulation_steps=3,
            patience_steps=100,
        ),
        **get_training_utils(
            args.model_type,
            corpus=fetch_and_load_corpus('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
        ),
    )

    study = optuna.create_study(direction='minimize')
    study.optimize(objective(train), n_trials=args.trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(study.best_trial.params, f, indent=4)

    print(f'Best trial parameters saved to {args.output}')