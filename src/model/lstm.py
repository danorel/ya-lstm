import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.25,
        device: torch.device = torch.device('cpu')
    ) -> None:
        super(LSTMCell, self).__init__()

        self.device = device
        self.dropout = nn.Dropout(dropout).to(self.device)

        self.input_to_forget = nn.Linear(input_size, hidden_size).to(self.device)
        self.input_to_input = nn.Linear(input_size, hidden_size).to(self.device)
        self.input_to_cell = nn.Linear(input_size, hidden_size).to(self.device)
        self.input_to_output = nn.Linear(input_size, hidden_size).to(self.device)

        self.hidden_to_forget = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.hidden_to_input = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.hidden_to_cell = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.hidden_to_output = nn.Linear(hidden_size, hidden_size).to(self.device)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, c_prev, h_prev):
        x = x.to(self.device)
        c_prev = c_prev.to(self.device)
        h_prev = h_prev.to(self.device)

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

    def forward(self, x):
        x = x.to(self.device)

        if self._hidden_states is None:
            self._init_hidden_states(x.size(0))

        prev_hidden_states = self._hidden_states
        
        o = []
        for t in range(x.size(1)):
            i = x[:, t, :]
            next_hidden_states = []
            for k in range(self.lstm_size):
                lstm = self.lstm_cells[k]
                c_prev, h_prev = prev_hidden_states[k]
                c_next, h_next = lstm(i, c_prev, h_prev)
                h_next = self.dropout(h_next)
                next_hidden_states.append((c_next, h_next))
                i = h_next
            o.append(next_hidden_states[-1][1])
        o = torch.stack(o, dim=1).to(self.device)
        o = self.decoder(o)

        self._hidden_states = [(c.detach(), h.detach()) for c, h in next_hidden_states]
        
        return o