import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from src.constants.device import device
from src.constants.modelling import ArchitectureName


class GRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=True).to(device)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True).to(device)
        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h_prev):
        gate_x = self.x2h(x)
        gate_h = self.h2h(h_prev)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        reset_gate = F.sigmoid(i_r + h_r)
        input_gate = F.sigmoid(i_i + h_i)
        output_gate = F.tanh(i_n + (reset_gate * h_n))

        h_next = output_gate + input_gate * (h_prev - output_gate)

        return h_next


class GRU(nn.Module):
    name = ArchitectureName.GRU.value

    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        cells_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size).to(device)
        self.cells = nn.ModuleList(
            [
                GRUCell(
                    input_size=embedding_size if k == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for k in range(cells_size)
            ]
        )
        self.dropout = nn.Dropout(dropout).to(device)
        self.decoder = nn.Linear(hidden_size, output_size).to(device)
        self.init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.tensor):
        batch_size = x.size(0)
        sequence_size = x.size(1)

        h_t = Variable(torch.zeros(batch_size, self.hidden_size)).to(device)
        outs = Variable(torch.zeros(batch_size, sequence_size, self.hidden_size)).to(device)

        x = self.embedding(x)
        for t in range(sequence_size):
            i_t = x[:, t, :]
            for gru in self.cells:
                h_t = gru(i_t, h_t)
                i_t = h_t
            outs[:, t, :] = i_t
        out = self.decoder(self.dropout(outs))

        return out
