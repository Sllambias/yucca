import logging
import math
from typing import Union
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle
from sklearn.model_selection import KFold
from dataclasses import dataclass
from yucca.training.configuration.configure_paths import PathConfig


@dataclass
class SplitConfig:
    splits: Union[dict[dict[list[dict]]], None] = None  # Contains `{ method: { parameter_value: [splits] }}`
    method: str = None
    param: Union[int, float] = None

    def split(self):
        return self.splits[str(self.method)][self.param]

    def train(self, idx):
        return self.split()[idx]["train"]

    def val(self, idx):
        return self.split()[idx]["val"]

    def lm_hparams(self):
        return {"split_method": self.method, "split_param": self.param}


def get_split_config(method: str, param: Union[str, float, int], path_config: PathConfig):
    """
    Params:
        method: SplitMethods
        param: Int or float depending on method param
    """
    splits_path = join(path_config.task_dir, "splits.pkl")

    if isfile(splits_path):
        splits = load_pickle(splits_path)

        # Overwrite old splits
        if not isinstance(splits, dict):
            splits = {}

        if split_is_precomputed(splits, method, param):
            logging.warning(
                f"Reusing already computed split file which was split using the {method} method and parameter {param}."
            )
            return SplitConfig(splits, method, param)
        else:
            logging.warning("Generating new split since splits did not contain a split computed with the same parameters.")
    else:
        splits = {}

    if method not in splits.keys():
        splits[method] = {}

    file_names = get_file_names(path_config.train_data_dir)

    if method == "kfold":
        splits[method][param] = kfold_split(file_names, int(param))
    elif method == "simple_train_val_split":
        splits[method][param] = simple_split(file_names, float(param))
    else:
        assert split_is_precomputed(
            splits, method, param
        ), "you are using a custom split method, but the split.pkl does not contain such a split"

    split_cfg = SplitConfig(splits, method, param)
    save_pickle(splits, splits_path)
    return split_cfg


def split_is_precomputed(splits, method, param):
    return method in splits.keys() and param in splits[method].keys()


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
