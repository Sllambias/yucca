import logging
import math
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle
from sklearn.model_selection import KFold
from yucca.paths import yucca_preprocessed_data
from dataclasses import dataclass
from typing import Union


@dataclass
class SplitConfig:
    splits: Union[list[dict], None] = None
    k: int = None
    p: float = None

    def train(self, idx):
        return self.splits[idx]["train"]

    def val(self, idx):
        return self.splits[idx]["val"]

    def lm_hparams(self):
        return {"k": self.k, "p": self.p}


def get_split_config(
    train_data_dir: str,
    task: str,
    k: int = 5,
    p: float = None,
):
    """
    Params:
        k: k for k-folds
        p: fraction of data to use for val split

    Note:
        You can only provide one of `k` or `p`.
        - If `k` is provided we will split with `k-fold`.
        - If `p` is provided it determines the fraction of items used for the val split.
    """

    assert (k is not None and p is None) or (k is None and p is not None), "You can only provide one of `k` or `p`."
    if p is not None:
        assert 0 < p < 1, "`p` must be a number between 0 and 1 and determines the fraction of items used for the val split"
    if k is not None:
        assert k > 0
        assert isinstance(k, int), "`k` should be an integer"

    method = "kfold" if k is not None else "simple_train_val_split"

    splits_path = join(yucca_preprocessed_data, task, "splits.pkl")
    if isfile(splits_path):
        split_cfg = load_pickle(splits_path)
        if split_cfg_is_the_same(split_cfg, k, p):
            logging.warning(f"Reusing already computed split file which was split using the {method} method")
            return split_cfg
        else:
            logging.warning(
                "Generating new split_cfg since split_cfg was either the wrong type or was generated using a different `k` and `p`."
            )

    file_names = get_file_names(train_data_dir)

    if method == "kfold":
        splits = kfold_split(file_names, k)
    else:
        splits = simple_split(file_names, p)

    split_cfg = SplitConfig(splits, k, p)
    save_pickle(split_cfg, splits_path)
    return split_cfg


def split_cfg_is_the_same(split_cfg, k, p):
    return isinstance(split_cfg, SplitConfig) and split_cfg.k == k and split_cfg.p == p


def kfold_split(file_names: list[str], k: int):
    assert k is not None
    kf = KFold(n_splits=k, shuffle=True)
    splits = []
    for train, val in kf.split(file_names):
        splits.append({"train": list(file_names[train]), "val": list(file_names[val])})
    return splits


def simple_split(file_names: list[str], p: float):
    assert p is not None
    assert 0 < p < 1
    np.random.shuffle(file_names)  # inplace
    num_val = math.ceil(len(file_names) * p)
    if num_val < 10:
        logging.warning("The validation split is very small. Consider using a higher `p`.")
    return [{"train": list(file_names[num_val:]), "val": list(file_names[:num_val])}]


def get_file_names(train_data_dir):
    file_names = subfiles(train_data_dir, join=False, suffix=".npy")
    if not file_names:
        file_names = subfiles(train_data_dir, join=False, suffix=".npz")
        if file_names:
            logging.warning("Only found compressed (.npz) files. This might increase runtime.")

    assert file_names, f"Couldn't find any .npy or .npz files in {train_data_dir}"
    return np.array(file_names)
