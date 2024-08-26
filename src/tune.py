import argparse
import json
import optuna
import torch

from pathlib import Path

from src.train import train
from src.corpus_loader import fetch_and_load_corpus

def objective(use_tensorboard: bool, num_workers: int = 1):
    def experiment(trial):
        base_hyperparameters = {
            'num_workers': num_workers,
            'epochs': 1,
            'max_steps': 101,
            'steps_patience': 10,
            'loss_patience': 0.001
        }
        tune_hyperparameters = {
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'cells_size': trial.suggest_int('cells_size', 1, 3),
            'hidden_size': trial.suggest_int('hidden_size', 128, 512),
            'sequence_size': trial.suggest_int('sequence_size', 16, 64),
            'batch_size': trial.suggest_int('batch_size', 128, 1024),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
            'accumulation_steps': trial.suggest_int('accumulation_steps', 1, 5)
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        corpus = fetch_and_load_corpus('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
        accuracy = train(
            device,
            corpus,
            name='lstm',
            hyperparameters={
                **base_hyperparameters,
                **tune_hyperparameters
            },
            use_tensorboard=use_tensorboard
        )
        return accuracy
    
    return experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--use_tensorboard', action='store_true',
                        help="Enable tensorboard metrics collection")
    parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of workers')
    parser.add_argument('--trials', type=int, required=True,
                    help='Amount of hyperparameter search iterations (e.g., 1, 4, 8, etc.)')
    parser.add_argument('--output', type=str, required=False, default='models/lstm/best_trial.json',
                        help='Output file to store the best trial parameters (e.g., "models/lstm/best_trial.json")')

    args = parser.parse_args()

    print(f"Tuning config:\n\ttensorboard = {args.use_tensorboard}\n\tworkers = {args.num_workers}")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        objective(
            use_tensorboard=args.use_tensorboard,
            num_workers=args.num_workers
        ),
        n_trials=args.trials
    )

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(study.best_trial.params, f, indent=4)

    print(f'Best trial parameters saved to {args.output}')