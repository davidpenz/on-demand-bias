import torch
import torch.nn as nn


def general_weight_init(m):
    # gain = nn.init.calculate_gain("tanh")
    # if type(m) == nn.Linear:
    #     torch.nn.init.xavier_normal_(m.weight, gain=gain)
    #     torch.nn.init.constant_(m.bias, 0.01)
    if type(m) == nn.Linear:
        if m.weight.requires_grad:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                torch.nn.init.constant_(m.bias, 0)

    elif type(m) == nn.Embedding:
        if m.weight.requires_grad:
            torch.nn.init.normal_(
                m.weight, std=0.1 / m.weight.shape[-1]
            )  # std suggested by https://dl.acm.org/doi/10.1145/3523227.3548486 (see Appendix)
