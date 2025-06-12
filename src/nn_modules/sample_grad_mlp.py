from typing import Union

import torch
from torch import nn
from collections import OrderedDict

from src.nn_modules.parallel import Parallel, ParallelMode
from src.nn_modules.gradient_reversal import GradientScalerLayer
from src.nn_modules.polylinear import PolyLinear


class GradAmpPolyLinear(PolyLinear):
    def __init__(
        self,
        layer_config: list,
        activation_fn: Union[str, nn.Module] = nn.ReLU(),
        output_fn=None,
        input_dropout=None,
        l1_weight_decay=None,
    ):
        """
        Helper module to easily create multiple linear layers and pass an
        activation through them
        :param layer_config: A list containing the in_features and out_features for the linear layers
                             Example: [100,50,2] would create two linear layers: Linear(100, 50) and Linear(50, 2),
                             whereas the output of the first layer is used as input for the second layer
        :param activation_fn: The activation function to use between layers
        :param output_fn: (optional) The function to apply on the output, e.g. softmax
        :param input_dropout: A possible dropout to apply to the input before passing it through the layers
        :param l1_weight_decay: Additional L1 weight normalization to induce sparsity in the layers
        """
        super().__init__(
            layer_config, activation_fn, output_fn, input_dropout, l1_weight_decay
        )

        self.gssl = GradientScalerLayer(1)

    def forward(self, x, sample_weights):
        x = self.layers(x)
        x = self.gssl(x, sample_weights)
        return x
