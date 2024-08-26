import torch.nn as nn
import typing as t

from .lstm import LSTM 
from .gru import GRU

model_selector: t.Dict[str, nn.Module] = {
    model.name: model
    for model in [LSTM, GRU]
}