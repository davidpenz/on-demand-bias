import os
from typing import Union, List, Tuple, Dict
import scipy.sparse as sp

import torch
from torch.utils.data import DataLoader
from joblib.externals.loky.backend import get_context

from src.config.data_paths import get_data_path
from src.data.FairDataset import FairDataset

from src.data.BaseDataset import BaseDataset
from src.data.user_feature import FeatureDefinition


def sparse_scipy_to_tensor(matrix):
    return torch.sparse_coo_tensor(*sparse_scipy_to_tensor_params(matrix))


def sparse_scipy_to_tensor_params(matrix):
    # sparse tensor multiprocessing in dataloaders is not supported,
    # therefore we will create the sparse tensor only in training loop
    m = matrix.tocoo()
    indices = torch.stack([torch.tensor(m.row), torch.tensor(m.col)])
    return indices, m.data, m.shape


def sparse_tensor_to_sparse_scipy(tensor: torch.Tensor):
    return sp.coo_matrix((tensor._values(), tensor._indices()), shape=tensor.shape)


def train_collate_fn(data):
    # data must not be batched (not supported by PyTorch layers)
    indices, user_data, item_data, targets = data
    user_data = sparse_scipy_to_tensor_params(user_data)
    item_data = sparse_scipy_to_tensor_params(item_data)
    targets = torch.tensor(targets)
    return indices, user_data, item_data, targets


def train_collate_fn_fair(data):
    *data, traits = data
    return *train_collate_fn(data), torch.tensor(traits)


def get_dataset(
    dataset: str,
    fold: int,
    dataset_type: str,
    splits: List[str],
    user_features: List,
):
    datasets = {}
    user_features = [FeatureDefinition(**d) for d in user_features]
    data_path = get_data_path(dataset)
    for split in splits:
        dataset = FairDataset(
            data_dir=os.path.join(data_path, str(fold)),
            split=split,
            features=user_features,
            transform=None,
        )
        datasets[split] = dataset
    return datasets


def get_dataloader(
    datasets: dict[str, BaseDataset],
    batch_size: Union[int, None] = 64,
    eval_batch_size: Union[int, None] = 64,
    n_workers: int = 0,
    shuffle_train=True,
) -> dict[str, DataLoader]:

    dataloader_dict = {}
    for split, dataset in datasets.items():
        is_train_split = split == "train"
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size if split == "train" else eval_batch_size,
            num_workers=n_workers,
            shuffle=is_train_split and shuffle_train,
            pin_memory=True,
            # multiprocessing_context=(get_context("loky") if n_workers > 0 else None),
        )
        dataloader_dict[split] = loader

    return dataloader_dict


def get_dataset_and_dataloader(
    config,
) -> Tuple[Dict[str, BaseDataset], Dict[str, DataLoader]]:
    datasets = get_dataset(**config["dataset_config"])
    dataloader = get_dataloader(datasets, **config["trainer"]["dataloader"])
    return datasets, dataloader
