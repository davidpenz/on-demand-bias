from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Tuple, Dict, Union


class BaseRecModel(nn.Module, ABC):
    @abstractmethod
    def build_from_config(self):
        pass

    @abstractmethod
    def calc_loss(
        self, logits, targets
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass


class AdvBaseRecModel(BaseRecModel):
    @abstractmethod
    def calc_adv_losses(self, adv_pred, adv_targets):
        pass

    @abstractmethod
    def build_adv_losses(self, adversary_config):
        pass

    @abstractmethod
    def build_adv_modules(self, adversary_config):
        pass


class VaeRecModel(ABC):
    @abstractmethod
    def encoder_forward(self, x: torch.Tensor):
        pass

    @abstractmethod
    def encode_user(self, x: torch.Tensor) -> torch.Tensor:
        pass
