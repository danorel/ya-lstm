import argparse
import math
import torch

from src.core.corpus_loader import fetch_and_load_corpus
from src.core.train import (
    ModelArchitecture,
    ModelHyperparameters,
    ModelTrainingConfig,
    get_training_utils,
    make_trainer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTM-like model on a specified text corpus.")
    
    parser.add_argument('--model_name', type=str, required=True, choices=['lstm', 'gru'],
                    help='Model to use as a basis for text generation (e.g., "lstm")')
    parser.add_argument('--model_type', type=str, required=True, choices=['character', 'word'],
                        help='Specify whether to train on a character-level or word-level model.')
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--use_profiler', action='store_true',
                        help="Enable profiling with torch.utils.bottleneck")
    parser.add_argument('--use_tensorboard', action='store_true', 
                        help="Enable tensorboard metrics collection")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for training.')
    parser.add_argument('--cells_size', type=int, default=2,
                        help='The number of LSTM layers.')
    parser.add_argument('--embedding_size', type=int, default=128,
                        help='The size of embedding layers.')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The size of hidden/context layers.')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Number of samples in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate parameter of LSTM optimizer (Adam is a default setting).')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma parameter of LSTM optimizer\'s scheduler (Adam is a default setting) which serves for training stabilization.')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay parameter of LSTM optimizer (Adam is a default setting) which serves for weights normalization to avoid overfitting.')
    parser.add_argument('--grad_clip_norm', type=float, default=5.0,
                        help='Gradient clipping which serves for weights update normalization to avoid gradient explosion and overflow.')
    parser.add_argument('--max_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--max_steps', type=float, default=+math.inf,
                        help='Number of training steps within the epoch.')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Accumulations steps per LSTM optimizer to make backward propagation pass (Adam is a default setting).')
    parser.add_argument('--patience_steps', type=int, default=100,
                        help='Patience steps to verify whether model has not stopped learning.')
    
    args = parser.parse_args()

    print(f"Training config:\n\tprofiler = {args.use_profiler}\n\ttensorboard = {args.use_tensorboard}\n")

    train = make_trainer(
        args.model_name,
        args.model_type,
        use_tensorboard=args.use_tensorboard,
        training_config=ModelTrainingConfig(
            max_epochs=(1 if args.use_profiler else args.max_epochs),
            max_steps=(100 if args.use_profiler else args.max_steps),
            accumulation_steps=args.accumulation_steps,
            patience_steps=args.patience_steps,
        ),
        **get_training_utils(
            args.model_type,
            corpus=fetch_and_load_corpus(args.url)
        ),
    )

    train(
        hidden_architecture=ModelArchitecture(
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            cells_size=args.cells_size,
            dropout=args.dropout
        ),
        hyperparameters=ModelHyperparameters(
            sequence_size=args.sequence_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm
        )
    )