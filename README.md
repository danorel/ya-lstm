# ya-lstm

ya-lstm is a Python project which is dealing with LSTM re-implementation. The idea is to understand the model details and how character-level language modelling can be trained via LSTM and GRU.

## Installation & Contributing

Use the [DEVELOPER.md](./DEVELOPER.md) guide to run or contribute to the project.

## Usage

1. Train LSTM agent on **default** shakespeare dataset with **default** hyperparameters:

```bash
python -m src.train --name lstm
```

2. Train LSTM agent on **custom** dataset (probably, it can be any .txt file) with **default** hyperparameters:

```bash
python -m src.train --name lstm --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

3. Train LSTM agent on **default** shakespeare dataset with the **custom** available and tunable hyperparameters: 'epochs', 'dropout', 'sequence_size', 'batch_size', 'learning_rate', and 'weight_decay':

```bash
python -m src.train --name lstm --epochs 5 --sequence_size 32 --dropout 0.3 --batch_size 256 --learning_rate 0.0001 --weight_decay 0.0001
```

4. Evaluate LSTM agent via generation sampling on **custom** dataset with **pre-trained** model hyperparameters:

```bash
python -m src.eval --name lstm --prompt_text 'Forecasting for you' --output_size 100
```

5. Tune LSTM agent via Optuna:

```bash
python -m src.tune
```

## Experiments

1. Trained a character-level language model via GRU model:

```bash
python -m src.train --name gru --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt --epochs 5 --dropout 0.25 --learning_rate 0.001 --sequence_size 64
```

2. Prompt on a pre-trained character-level language model via GRU model:

```bash
python -m src.eval --name gru --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt --sequence_size 64 --prompt_text 'hello, my darling, my name is lord orvald and i am fond of staring at your' --output_size 128
```

## License

[MIT](./LICENSE)