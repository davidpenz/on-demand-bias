# %%
from typing import Union, List

import torch
from torch import nn
import torch.nn.functional as F

from src.config_classes.adv_config import AdvConfig
from src.nn_modules.adversary import Adversary
from src.nn_modules.mult_dae import MultDAE
from src.nn_modules.parallel import Parallel, ParallelMode
from src.nn_modules.polylinear import PolyLinear

# %%
a = torch.rand(4, 6)
x = torch.bernoulli(a)
x_neg = 1 - x
r = torch.rand(4, 6)
r_act = torch.bernoulli(r)
r_pos, r_neg = r_act * x, r_act * x_neg
x_obf = x + r_neg - r_pos


# %%
class BernoulliActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
        out = self.tanh(x)
        out = torch.bernoulli(torch.abs(out))
        out = mask * out
        return out


class ObfuscationLayer(nn.Module):
    def __init__(self, size_input, item_ster) -> None:
        super().__init__()
        self.linear = nn.Linear(size_input, size_input)
        self.fun_act = nn.Tanh()
        self.item_ster = item_ster

    def forward(self, x):
        """_summary_

        Args:
            x (torch.tensor): batch_size, n_items

        Returns:
            x_obf(torch.tensor): batch_size, n_items
            x_ster(torch.tensor): batch_size, n_items
        """
        x_neg = 1 - x
        x_ster = self.fun_act(self.linear(x))
        x_ster_act = torch.bernoulli(torch.abs(x_ster))
        # Binary masks
        x_ster_pos, x_ster_neg = x_ster_act * x, x_ster_act * x_neg
        # Combining
        x_obf = x + x_ster_pos - x_ster_neg
        return x_obf, x_ster
