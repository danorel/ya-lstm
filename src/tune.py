import optuna

from src.train import train
from src.corpus_loader import fetch_and_load_corpus

def objective(trial):
    base_hyperparameters = {
        'epochs': 1,
        'max_steps': 128
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
    corpus = fetch_and_load_corpus('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
    accuracy = train(corpus, name='lstm', hyperparameters={
        **base_hyperparameters,
        **tune_hyperparameters
    })
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=16)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)