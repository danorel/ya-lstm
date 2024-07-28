import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(LSTMCell, self).__init__()
    
    def forward(self, x, h, c):
        pass
        

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

    def forward(self, x, h):
        pass

    def init_weights(self, batch_size: int = 1):
        pass