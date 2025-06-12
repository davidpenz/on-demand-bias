import numpy as np
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from enum import StrEnum, auto

from src.config_classes.adv_config import AdvConfig
from src.config_classes.atk_config import AtkConfig


class GeneralizedCrossEntropyLoss(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_

    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`


    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, alpha: float = 0.7, weight=[0.5, 0.5]) -> None:
        super().__init__()
        self.q = alpha
        self.epsilon = 1e-9
        self.softmax = nn.Softmax(dim=1)
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.weight * (self.softmax(input))
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon

        loss = (1 - p**self.q) / self.q

        return torch.mean(loss)


loss_fn_lookup = {
    "mse": MSELoss(),
    "mae": L1Loss(),
    "ce": CrossEntropyLoss(),
    "gce": GeneralizedCrossEntropyLoss(),
}


class AdvLosses(nn.Module):
    """
    Loss function that evaluates the performance of adversarial networks.
    """

    def __init__(self, adv_config: List[Union[AdvConfig, AtkConfig]]):
        super().__init__()
        self.adv_config = adv_config

    def _calc_loss_for_adv_group(
        self,
        config: Union[AdvConfig, AtkConfig],
        inputs: List[torch.Tensor],
        targets: torch.Tensor,
    ):
        individual_results = [
            self._calc_loss_for_single_adv(config, inp, targets) for inp in inputs
        ]
        return sum(individual_results) / len(individual_results)

    @staticmethod
    def _calc_loss_for_single_adv(
        config: Union[AdvConfig, AtkConfig], inputs: torch.Tensor, targets: torch.Tensor
    ):
        if config.loss_fn in ["mse", "mae"]:
            targets = torch.reshape(targets, (-1, 1))
        if config.loss_fn == "ce":
            weights = torch.Tensor(config.loss_class_weights).to(inputs.get_device())
            loss_fn = (
                CrossEntropyLoss(weight=weights)
                if config.loss_class_weights
                else CrossEntropyLoss()
            )

            return loss_fn(inputs, targets)
        if config.loss_fn == "gce":
            weights = torch.Tensor(config.loss_class_weights).to(inputs.get_device())
            loss_fn = (
                GeneralizedCrossEntropyLoss(alpha=config.loss_gce_alpha, weight=weights)
                if config.loss_gce_alpha
                else CrossEntropyLoss()
            )
            return loss_fn(inputs, targets)
        return loss_fn_lookup[config.loss_fn](inputs, targets)

    def forward(self, inputs: List[List[torch.Tensor]], targets: List[torch.Tensor]):
        """
        Calculates the losses for the different adversaries in the different adversarial groups.

        :param inputs: A list of lists of tensors, where the tensors are the results of the individual adversaries
                       and the sublists are the results for a group of adversaries.
        :param targets: A list of tensors, where each tensor is the target value an adversary should achieve
        """
        # calculate individual losses
        # already re-weight losses here, as we also want to report the new loss, rather than the original one
        losses = {
            cfg.group_name: self._calc_loss_for_adv_group(
                cfg,
                inp,
                tar.to(
                    dtype=torch.int64 if cfg.type == "categorical" else torch.float32
                ),
            )
            * cfg.loss_weight
            for cfg, inp, tar in zip(self.adv_config, inputs, targets)
        }

        return sum(losses.values()), losses


class MultVAELoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_neg = CrossEntropyLoss()

    def forward(
        self, logits, targets, mu, logvar, anneal_cap, n_update, total_anneal_steps
    ):
        z, mu, logvar = logits
        neg_ll = self.loss_neg(z, targets)
        kl_div = self._calc_KL_div(mu, logvar, anneal_cap, n_update, total_anneal_steps)
        loss = neg_ll + kl_div
        loss_dict = {"nll": neg_ll, "KL": kl_div}

        return loss, loss_dict

    def _calc_KL_div(self, mu, logvar, anneal_cap, n_update, total_anneal_steps):
        """
        Calculates the KL divergence of a multinomial distribution with the generated
        mean and std parameters
        """
        # Calculation for multinomial distribution with multivariate normal and standard normal distribution based on
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # Mean is used as we may have different batch sizes, thus possibly have different losses throughout training
        self.n_update += 1
        if self.total_anneal_steps > 0:
            anneal = min(anneal_cap, 1.0 * n_update / total_anneal_steps)
        else:
            anneal = anneal_cap
        kl_div = 0.5 * torch.mean(
            torch.sum(-logvar + torch.exp(logvar) + mu**2 - 1, dim=1) * anneal
        )

        return kl_div


class VAELoss(nn.Module):
    def __init__(self, beta=None, beta_cap=0.5, beta_steps=2000, beta_patience=5):
        """
        :param beta: if provided, the beta value will be kept at this value
        :param beta_cap: maximum value beta can reach
        :param beta_steps: maximum number of beta annealing steps
        :param beta_patience: number of steps with no improvement after which beta annealing should be halted
        """
        super().__init__()

        self.beta = beta
        self.beta_cap = beta_cap
        self.beta_steps = beta_steps
        self._curr_beta = 0

        if beta is not None:
            self._curr_beta = beta

        # Parameters for beta annealing
        self.patience = beta_patience
        self._n_steps_wo_increase = 0
        self._best_score = -np.inf

    def forward(self, logits: torch.Tensor, KL: float, y: torch.Tensor):
        prob = F.log_softmax(logits, dim=1)

        neg_ll = -torch.mean(torch.sum(prob * y, dim=1))
        weighted_KL = self._curr_beta * KL
        loss = neg_ll + weighted_KL

        return loss, {"nll": neg_ll, "KL": weighted_KL}

    def beta_step(self, score):
        """
        Performs the annealing procedure for the beta parameter
        Described in "Variational Autoencoders for Collaborative Filtering", Section 2.2.2
        :param score: The score used to determine whether to keep increasing the beta parameter
        :return: The current beta parameter, either updated or still from the previous call
        """
        if self.beta is not None:
            return self._curr_beta

        if self._n_steps_wo_increase > self.patience:
            return self._curr_beta

        # Even if validation score does not improve, we will still increase beta
        if self._best_score > score:
            self._n_steps_wo_increase += 1
        else:
            self._best_score = score
            self._n_steps_wo_increase = 0

        self._curr_beta += self.beta_cap / self.beta_steps
        self._curr_beta = min(self.beta_cap, self._curr_beta)
        return self._curr_beta


class StereoVAELoss(VAELoss):
    def __init__(
        self,
        item_ster_values=None,
        sigma=0.01,
        beta=None,
        beta_cap=0.5,
        beta_steps=2000,
        beta_patience=5,
    ):
        """
        :param item_ster_values: stereotypicality values
        :param beta: if provided, the beta value will be kept at this value
        :param beta_cap: maximum value beta can reach
        :param beta_steps: maximum number of beta annealing steps
        :param beta_patience: number of steps with no improvement after which beta annealing should be halted
        """
        super().__init__(beta, beta_cap, beta_steps, beta_patience)
        self.sigma = sigma
        self.item_ster = item_ster_values

    def forward(
        self, logits: torch.Tensor, KL: float, y: torch.Tensor
    ):  # ,class_label:torch.Tensor):
        loss_vae, loss_dict = super().forward(logits, KL, y)
        loss_stereo = self._calculate_stereo(logits, y)  # , class_label)
        loss = loss_vae + self.sigma * loss_stereo
        loss_dict.update({"loss_stereo": loss_stereo})
        return loss, loss_dict

    def _calculate_stereo(self, logits, x):  # , class_label):
        """_summary_

        Args:
            logits (torch.tensor): batch_size * n_items
            x (torch.tensor): batch_size * n_items
            gender_label (torch.tensor): batch_size * 1
        """
        # x_item_ster= self.item_ster * x #* class_label

        x_ster = logits * self.item_ster * x
        pos_mask = torch.where(x_ster >= 0, 1.0, 0.0)
        neg_mask = torch.where(x_ster < 0, 1.0, 0.0)

        x_ster_pos = torch.sum(torch.abs(x_ster * pos_mask), dim=1)
        x_ster_neg = torch.sum(torch.abs(x_ster * neg_mask), dim=1)

        return torch.abs(x_ster_pos - x_ster_neg).mean()

    def _calculate_stereo_prob(self, logits, x):  # , class_label):
        """_summary_

        Args:
            logits (torch.tensor): batch_size * n_items
            x (torch.tensor): batch_size * n_items
            gender_label (torch.tensor): batch_size * 1
        """
        # x_item_ster= self.item_ster * x #* class_label

        x_ster = self.item_ster * x
        pos_mask = torch.where(x_ster >= 0, 1.0, 0.0)
        neg_mask = torch.where(x_ster < 0, 1.0, 0.0)

        x_ster_pos = torch.sum(torch.abs(x_ster * pos_mask), dim=1)
        x_ster_neg = torch.sum(torch.abs(x_ster * neg_mask), dim=1)

        return torch.abs(x_ster_pos - x_ster_neg).mean()
