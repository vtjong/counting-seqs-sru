import torch.nn as nn
from .sru import SRU

class CountingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_sru_layers=5):
        super(CountingModel, self).__init__()
        self.sru = SRU(input_size, hidden_size, num_sru_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.sru(x)
        out = self.fc(out)
        return out
