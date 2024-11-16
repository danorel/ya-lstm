import pathlib

import torch
from torch import nn

from src.architecture import GRU, LSTM
from src.constants.device import device
from src.constants.metadata import MODEL_ARCHIVE_DIR, MODEL_REPOSITORY_DIR
from src.modelling.common.model_arguments import Metadata

model_archive_dir = pathlib.Path(MODEL_ARCHIVE_DIR)
model_archive_dir.mkdir(parents=True, exist_ok=True)

select_architecture: dict[str, nn.Module] = {architecture.name: architecture for architecture in (LSTM, GRU)}


def load_model_from_archive(metadata: Metadata, version: str = "latest") -> nn.Module:
    model_file = model_archive_dir / metadata.architecture_name / metadata.modelling_name / version / "model.pt"
    if not model_file.exists():
        raise RuntimeError("Not found pre-trained LSTM-based model")
    model: nn.Module = torch.load(model_file)
    model.to(device)
    return model


def load_model_to_repository(model: nn.Module, metadata: Metadata, version: str, example: torch.tensor) -> nn.Module:
    model_repository_dir = pathlib.Path(MODEL_REPOSITORY_DIR)
    model_file_dir = model_repository_dir / metadata.architecture_name / metadata.modelling_name / version
    model_file_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, model_file_dir / "model.onnx", input_names=["input"], output_names=["output"])
