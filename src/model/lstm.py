import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.25,
        device: torch.device = torch.device('cpu')
    ):
        super(LSTMCell, self).__init__()

        self.device = device

        self.dropout = nn.Dropout(dropout).to(self.device)
        self.weight = nn.Parameter(torch.Tensor(4 * hidden_size, input_size + hidden_size)).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(4 * hidden_size)).to(self.device)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, c_prev, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gate_values = combined.matmul(self.weight.t()) + self.bias

        forget_gate, input_gate, cell_gate, output_gate = torch.split(gate_values, gate_values.size(1) // 4, dim=1)
        
        forget_gate = torch.sigmoid(forget_gate)
        input_gate = torch.sigmoid(input_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)

        return c_next, self.dropout(h_next)

class LSTM(nn.Module):
    name = 'lstm'

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.25,
        lstm_size: int = 1,
        device: torch.device = torch.device('cpu')
    ):
        super(LSTM, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, dropout) 
            for i in range(lstm_size)
        ])
        self.decoder = nn.Linear(hidden_size, output_size).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

        self._init_weights()

        self._hidden_states = None
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_hidden_states(self, batch_size: int = 1):
        self._hidden_states = [
            (
                torch.zeros(batch_size, self.hidden_size).to(self.device),
                torch.zeros(batch_size, self.hidden_size).to(self.device)
            )
            for _ in range(self.lstm_size)
        ]

    def reset_hidden_states(self):
        self._hidden_states = None

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        sequence_size = x.size(1)
        batch_size = x.size(0)
        hidden_size = self.hidden_size

        if self._hidden_states is None:
            self._init_hidden_states(batch_size)

        prev_hidden_states = self._hidden_states
        o = torch.zeros(batch_size, sequence_size, hidden_size, device=self.device)

        for t in range(x.size(1)):
            i = x[:, t, :]
            next_hidden_states = []
            for k in range(self.lstm_size):
                lstm = self.lstm_cells[k]
                c_prev, h_prev = prev_hidden_states[k]
                c_next, h_next = lstm(i, c_prev, h_prev)
                next_hidden_states.append((c_next, h_next))
                i = h_next
            prev_hidden_states = next_hidden_states
            o[:, t, :] = next_hidden_states[-1][1]

        o = self.decoder(o)

        self._hidden_states = [(c.detach(), h.detach()) for c, h in next_hidden_states]
        
        return o