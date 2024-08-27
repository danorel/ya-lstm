import pathlib
import torch
import torch.nn as nn
import typing as t

from src.constants import MODEL_ARCHIVE_DIR, MODEL_REPOSITORY_DIR
from .lstm import LSTM 
from .gru import GRU

model_selector: t.Dict[str, nn.Module] = {
    model.name: model
    for model in [LSTM, GRU]
}

def load_model_from_archive(device: torch.device, name: str) -> nn.Module:
    model_archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR)
    model_archive_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_archive_dir / name / '2' / '200' / 'model.pt'
    if not model_file.exists():
        raise RuntimeError("Not found pre-trained RNN-based model")
    model: nn.Module = torch.load(model_file)
    model.to(device)
    return model


def load_model_to_repository(model: nn.Module, example_embedding: torch.tensor, name: str, version: int) -> nn.Module:
    model_repository_dir = pathlib.Path(MODEL_REPOSITORY_DIR)
    model_file_dir = model_repository_dir / name / f"{version}"
    model_file_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_embedding,
        model_file_dir / "model.onnx",
        input_names=['input'],
        output_names=['output']
    )