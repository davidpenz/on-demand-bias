import torch.nn as nn
from src.nn_modules.quant_layer import Quantizer


class Mask(nn.Module):
    def __init__(self, encoder_dims, mask_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(encoder_dims[0], mask_hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mask_hidden_dim, encoder_dims[0])
        self.out = nn.Sigmoid()
        self.bin = Quantizer()

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        out = self.bin(self.out(self.fc2(hidden)))
        return out, hidden
