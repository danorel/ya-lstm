# ya-lstm

ya-lstm is a Python project which is dealing with LSTM re-implementation. The idea is to understand the model details and how character-level language modelling can be trained via LSTM and GRU.

## Installation & Contributing

Use the [DEVELOPER.md](./DEVELOPER.md) guide to run or contribute to the project.

## Usage

1. Train LSTM agent on **default** shakespeare dataset with **default** hyperparameters:

```bash
python \
    -m src.train \
    --name lstm
```

2. Train LSTM agent on **custom** dataset (probably, it can be any .txt file) with **default** hyperparameters:

```bash
python \
    -m src.train \
    --name lstm \
    --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

3. Train LSTM agent on **custom** shakespeare dataset with the **custom** available and tunable hyperparameters: 'use_profiler', 'use_tensorboard', 'num_workers', 'name', 'url', 'epochs', 'max_steps', 'dropout', 'cells_size', 'hidden_size', 'sequence_size', 'batch_size', 'learning_rate', and 'weight_decay':

```bash
python \
    -m src.train \
    --model_name gru \
    --model_type word \
    --use_tensorboard \
    --epochs 5 \
    --max_steps 1000 \
    --dropout 0.1 \
    --cells_size 3 \
    --embedding_size 256 \
    --hidden_size 512 \
    --sequence_size 64 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --accumulation_steps 3
```

4. Evaluate LSTM agent via generation sampling on **custom** dataset with **pre-trained** model hyperparameters:

```bash
python \
    -m src.eval \
    --model_name lstm \
    --model_type word \
    --prompt_text 'Hello my dear darling and princess, ' \
    --sequence_size 64 \
    --output_size 128
```

5. Tune LSTM agent via Optuna:

```bash
python \
    -m src.tune \
    --model_name lstm \
    --model_type word \
    --trials 16
```

6. Profile LSTM agent via torch.utils.bottleneck:

```bash
python \
    -m torch.utils.bottleneck src/train.py \
    --model_name lstm \
    --model_type word \
    --use_profiler \
    --num_workers 0 \
    --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    --accumulation_steps 5
```

6. Track metrics of LSTM agent during training via Tensorboard:

```bash
python \
    -m src.train \
    --model_name lstm \
    --model_type word \
    --use_tensorboard
```

## Experiments

1. Trained a character-level language model via GRU model:

```bash
python \
    -m src.train \
    --model_name gru \
    --model_type word \
    --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt \
    --epochs 5 \
    --dropout 0.25 \
    --learning_rate 0.001 \
    --sequence_size 64
```

2. Prompt on a pre-trained character-level language model via GRU model:

```bash
python \
    -m src.eval \
    --model_name gru \
    --model_type word \
    --url https://github.com/karpathy/char-lstm/blob/master/data/tinyshakespeare/input.txt \
    --sequence_size 64 \
    --prompt_text 'hello, my darling, my name is lord orvald and i am fond of staring at your' \
    --output_size 128
```

## Deployment

1. Export LSTM model from PyTorch to ONNX format:

```bash
python \
    -m src.export \
    --model_name lstm \
    --model_type word \
    --sequence_size 64
```

2. Serve ONNX model via Triton Inference Server:

```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ./model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models
```

## License

[MIT](./LICENSE)