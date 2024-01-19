import torch
import torch.nn as nn

from .sru_cell import SRUCell

class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0, layer_norm=False):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.layers = nn.Sequential()

        # input layer
        self.layers.append(SRUCell(input_size, hidden_size, layer_norm=layer_norm, dropout=dropout if num_layers!=1 else 0))

        # hidden layers
        for i in range(1, num_layers):
            dropout = dropout if i + 1 != num_layers else 0
            layer = SRUCell(hidden_size, hidden_size, layer_norm, dropout)
            self.layers.append(layer)

    def forward(self, x):
        # init cell state
        c0 = torch.zeros(x.size(1), self.hidden_size)

        # input layer
        h, c = self.layers[0](x, c0)

        # hidden layers
        for layer in self.layers[1:]:
            h, c = layer(h, c0)

        return h, c

