import math
import typing as t

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from src.constants.device import device
from src.constants.modelling import ArchitectureName


class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=True).to(device)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True).to(device)
        self._init_weights()

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, c_prev, h_prev):
        gates = self.x2h(x) + self.h2h(h_prev)

        forget_gate, input_gate, cell_gate, output_gate = gates.chunk(4, 1)

        forget_gate = F.sigmoid(forget_gate)
        input_gate = F.sigmoid(input_gate)
        cell_gate = F.tanh(cell_gate)
        output_gate = F.sigmoid(output_gate)

        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * F.tanh(c_next)

        return c_next, h_next


class LSTM(nn.Module):
    name = ArchitectureName.LSTM.value

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        cells_size: int = 1,
        dropout: float = 0.25,
        pretrained_embeddings: t.Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        self.cells = nn.ModuleList(
            [
                LSTMCell(
                    input_size=embedding_size if k == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for k in range(cells_size)
            ]
        )
        self.dropout = nn.Dropout(dropout).to(device)
        self.decoder = nn.Linear(hidden_size, output_size).to(device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.tensor):
        batch_size = x.size(0)
        sequence_size = x.size(1)

        c_t = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        h_t = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        outs = Variable(torch.zeros(batch_size, sequence_size, self.hidden_size)).to(device)

        x = self.embedding(x)
        for s_t in range(sequence_size):
            i_t = x[:, s_t, :]
            for idx, lstm in enumerate(self.cells):
                c_t, h_t = lstm(i_t, c_t, h_t)
                i_t = h_t
                if idx < len(self.cells) - 1:
                    i_t = self.dropout(i_t)
            outs[:, s_t, :] = i_t
        out = self.decoder(self.dropout(outs))

        return out
