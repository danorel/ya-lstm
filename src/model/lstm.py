import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.25
    ) -> None:
        super(LSTMCell, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.input_to_forget = nn.Linear(input_size, hidden_size)
        self.input_to_input = nn.Linear(input_size, hidden_size)
        self.input_to_cell = nn.Linear(input_size, hidden_size)
        self.input_to_output = nn.Linear(input_size, hidden_size)

        self.hidden_to_forget = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_input = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_cell = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, hidden_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, c_prev, h_prev):
        forget_gate = torch.sigmoid(self.input_to_forget(x) + self.hidden_to_forget(h_prev))
        input_gate = torch.sigmoid(self.input_to_input(x) + self.hidden_to_input(h_prev))
        cell_gate = torch.tanh(self.input_to_cell(x) + self.hidden_to_cell(h_prev))
        output_gate = torch.sigmoid(self.input_to_output(x) + self.hidden_to_output(h_prev))
        
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
        dropout: float = 0.25
    ):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm_cell = LSTMCell(input_size, hidden_size, dropout)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, c = None, h = None):
        if h is None:
            h = self.init_hidden_state(x.size(0))
        if c is None:
            c = self.init_hidden_state(x.size(0))

        o = []
        for pos in range(x.size(1)):
            c, h = self.lstm_cell(x[:, pos, :], c, h)
            o.append(self.dropout(self.decoder(h)))
        o = torch.stack(o, dim=1)
        
        return o, c, h

    def init_hidden_state(self, batch_size: int = 1):
        return torch.zeros(batch_size, self.hidden_size)