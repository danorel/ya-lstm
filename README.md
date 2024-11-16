# ya-lstm

ya-lstm is a Python project which is dealing with LSTM re-implementation. The idea is to understand the model details and how character-level language modelling can be trained via LSTM and GRU.

## Installation and Setup

### Set up Micromamba Environment

Ensure you have Micromamba installed. Then, create and activate the environment:

```bash
micromamba create -n ya-lstm -f env.yaml
micromamba activate ya-lstm
```

This will install Python 3.11 and all dependencies listed in `env.yaml`. If you add new dependencies you can sync up environment in such a way:

```bash
micromamba env update --file env.yaml
```

### Configure Development Tools

The `pyproject.toml` file is pre-configured with `black` for code formatting, `isort` for import sorting, `pytest` for testing, and `pylint` for code linting. These tools help maintain code quality and consistency.

Install `pre-commit` to apply code formatting and import sorting before each commit is done:

```bash
pre-commit install
pre-commit autoupdate
```

## Usage

### Running the application

#### Training phase

1. Train LSTM agent on **default** shakespeare dataset with **default** hyperparameters:

```bash
python \
    -m src.phase.train \
    --name lstm
```

2. Train LSTM agent on **custom** dataset (probably, it can be any .txt file) with **default** hyperparameters:

```bash
python \
    -m src.phase.train \
    --name lstm \
    --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

3. Train LSTM agent on **custom** shakespeare dataset with the **custom** available and tunable hyperparameters: 'use_profiler', 'use_tensorboard', 'num_workers', 'name', 'url', 'epochs', 'max_steps', 'dropout', 'cells_size', 'hidden_size', 'sequence_size', 'batch_size', 'learning_rate', and 'weight_decay':

```bash
python \
    -m src.phase.train \
    --architecture_name lstm \
    --modelling_name word_level \
    --use_tensorboard \
    --dropout 0.1 \
    --cells_size 5 \
    --embedding_size 128 \
    --hidden_size 256 \
    --sequence_size 64 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --max_epochs 5 \
    --max_steps 1000 \
    --accumulation_steps 3 \
    --patience_steps 100
```

4. Train a GRU agent via character-level language model:

```bash
python \
    -m src.phase.train \
    --architecture_name gru \
    --modelling_name word_level \
    --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt \
    --epochs 5 \
    --dropout 0.25 \
    --learning_rate 0.001 \
    --sequence_size 64
```

#### Optimization phase

1. Tune LSTM agent via Optuna:

```bash
python \
    -m src.phase.tune \
    --architecture_name lstm \
    --modelling_name word_level \
    --trials 16
```

2. Profile LSTM agent via torch.utils.bottleneck:

```bash
python \
    -m torch.utils.bottleneck src/train.py \
    --architecture_name lstm \
    --modelling_name word_level \
    --use_profiler \
    --num_workers 0 \
    --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    --accumulation_steps 5
```

3. Track metrics of LSTM agent during training via Tensorboard:

```bash
python \
    -m src.phase.train \
    --architecture_name lstm \
    --modelling_name word_level \
    --use_tensorboard
```

#### Evaluation phase

1. Evaluate LSTM agent via generation sampling on **custom** dataset with **pre-trained** model hyperparameters:

```bash
python \
    -m src.phase.prompt \
    --architecture_name lstm \
    --modelling_name word_level \
    --input_text 'Hello my dear darling and princess, ' \
    --sequence_size 64 \
    --output_size 32
```


2. Prompt on a pre-trained character-level language model via GRU model:

```bash
python \
    -m src.phase.prompt \
    --architecture_name gru \
    --modelling_name word_level \
    --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt \
    --sequence_size 64 \
    --input_text 'hello, my darling, my name is lord orvald and i am fond of staring at your' \
    --output_size 128
```

### Deployment of the application

1. Export LSTM model from PyTorch to ONNX format:

```bash
python \
    -m src.phase.export \
    --architecture_name lstm \
    --modelling_name word_level \
    --sequence_size 64
```

2. Serve ONNX model via Triton Inference Server:

```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models
```

### Testing and code quality

This project uses several tools to ensure code quality and reliability. These tools are configured in both pyproject.toml and .gitlab-ci.yml.

#### Code quality and auditing

**Linting**: Uses `pylint` for code quality analysis. To run locally:

```bash
pre-commit run pylint --all-files # or pylint src/
```

**Dependency auditing**: Uses `pip-audit` to check for known vulnerabilities in Python packages. Run:

```bash
pre-commit run pip-audit --all-files # or pip-audit
```

## License

[MIT](./LICENSE)