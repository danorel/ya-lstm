import argparse
import json
import optuna
import torch

from pathlib import Path

from src.train import train
from src.corpus_loader import fetch_and_load_corpus

def objective(trial):
    base_hyperparameters = {
        'epochs': 1,
        'max_steps': 101
    }
    tune_hyperparameters = {
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lstm_size': trial.suggest_int('lstm_size', 1, 3),
        'hidden_size': trial.suggest_int('hidden_size', 128, 512),
        'sequence_size': trial.suggest_int('sequence_size', 16, 64),
        'batch_size': trial.suggest_int('batch_size', 128, 1024),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
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
        }
    )
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--trials', type=int, required=True,
                    help='Amount of hyperparameter search iterations (e.g., 1, 4, 8, etc.)')
    parser.add_argument('--output', type=str, required=False, default='models/lstm/best_trial.json',
                        help='Output file to store the best trial parameters (e.g., "models/lstm/best_trial.json")')

    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(study.best_trial.params, f, indent=4)

    print(f'Best trial parameters saved to {args.output}')