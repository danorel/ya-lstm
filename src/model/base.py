import torch
import torch.nn as nn

from .lstm import LSTMCell

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.00)
    elif isinstance(m, LSTMCell):
        torch.nn.init.xavier_uniform_(m.weight_xh)
        torch.nn.init.xavier_uniform_(m.weight_hh)
        torch.nn.init.zeros_(m.bias_xh)
        torch.nn.init.zeros_(m.bias_hh)