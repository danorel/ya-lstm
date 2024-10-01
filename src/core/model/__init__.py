import pathlib
import torch
import torch.nn as nn
import typing as t

from src.core.constants import MODEL_ARCHIVE_DIR, MODEL_REPOSITORY_DIR
from .lstm import LSTM 
from .gru import GRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_selector: t.Dict[str, nn.Module] = {
    model.name: model
    for model in [LSTM, GRU]
}

def load_model_from_archive(model_name: str, model_type: str) -> nn.Module:
    model_archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR)
    model_archive_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_archive_dir / model_name / model_type / '1' / '150' / 'model.pt'
    if not model_file.exists():
        raise RuntimeError("Not found pre-trained LSTM-based model")
    model: nn.Module = torch.load(model_file)
    model.to(device)
    return model


def load_model_to_repository(model: nn.Module, example: torch.tensor, path: str, version: str) -> nn.Module:
    model_repository_dir = pathlib.Path(MODEL_REPOSITORY_DIR)
    model_file_dir = model_repository_dir / path / version
    model_file_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        model_file_dir / "model.onnx",
        input_names=['input'],
        output_names=['output']
    )