from torch.nn import Module
from src.nn_modules.quant_functional import QuantizerFunc


class Quantizer(Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantizerFunc.apply

    def forward(self, x):
        x = self.quant(x)
        return x

