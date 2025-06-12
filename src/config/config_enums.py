from typing import Optional
from enum import auto, Enum, StrEnum

import torch

from src.trainer import *


class DatasetSplitType(StrEnum):
    Random = auto()
    Temporal = auto()


class ModelInputType(StrEnum):
    ONE_HOT = auto()
    MULTI_HOT = auto()


class FeatureType(StrEnum):
    CATEGORICAL = auto()
    DISCRETE = auto()


class DatasetsEnum(StrEnum):
    """
    Enum to keep track of all the dataset available. Note that the name of the dataset  should correspond to a folder
    in data. e.g. ml-1m has a corresponding folder in /data
    """

    ml1m = auto()
    lfmdemobias = auto()
    ambar = auto()


class OptimEnum(StrEnum):
    adam = auto()
    adamw = auto()
    sgd = auto()
    adadelta = auto()
    rmsprop = auto()


class AlgorithmsEnum(StrEnum):
    multvae = auto()
    advxmultvae = auto()
    regmultvae = auto()
    maskmultvae = auto()


optim_class_choices = {
    OptimEnum.adam: torch.optim.Adam,
    OptimEnum.adamw: torch.optim.AdamW,
    OptimEnum.sgd: torch.optim.SGD,
    OptimEnum.adadelta: torch.optim.Adadelta,
    OptimEnum.rmsprop: torch.optim.RMSprop,
}
trainer_class_choices = {
    AlgorithmsEnum.multvae: AETrainer,
    AlgorithmsEnum.regmultvae: AdvAETrainer,
    AlgorithmsEnum.advxmultvae: AdvAETrainer,
    AlgorithmsEnum.maskmultvae: MaskAETrainer,
}


class AttackerEnum(StrEnum):
    vaeembeddingattacker = auto()


atk_trainer_class_choices = {AttackerEnum.vaeembeddingattacker: AttackNetTrainer}
