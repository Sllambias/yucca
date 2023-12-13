from dataclasses import dataclass
import logging
import math
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle
from enum import StrEnum, auto, asdict

from sklearn.model_selection import KFold
from yucca.paths import yucca_preprocessed_data

# We set this seed manually as multiple trainers might use this split,
# And we may not know which individual seed dictated the data splits
# Therefore for reproducibility this is fixed.
DEFAULT_SEED = 52189
allowed_split_methods = ["kfold", "simple"]


@dataclass
class SplitConfig:
    folds: list[dict]
    fold: int
    method: str
    seed: int
    k: int = None
    val_ratio: float = None

    def train(self):
        return self.folds[self.fold]["train"]

    def val(self):
        return self.folds[self.fold]["val"]

    def lm_hparams(self):
        return {"fold": self.fold, "method": self.method, "seed": self.seed, "k": self.k, "val_ratio": self.val_ratio}


def get_split_config(
    train_data_dir: str,
    task: str,
    fold: int = 0,
    method: str = "kfold",
    k: int = 5,
    val_ratio: float = 0.01,
    seed: int = DEFAULT_SEED,
):
    """
    If method is `SplitMethod.KFOLD` then a k argument must be provided
    If method is `SplitMethod.SIMPLE` a p argument must be provided
    """
    assert method in allowed_split_methods, f"The method {method} is not an allowed splitting method"

    splits_path = join(yucca_preprocessed_data, task, "splits.pkl")
    if isfile(splits_path):
        splits = load_pickle(splits_path)
        if isinstance(splits, SplitConfig):
            return splits

    files = load_files(train_data_dir)
    return perform_split(files, splits_path, fold, method, k, val_ratio, seed)


def perform_split(files: list[str], splits_path: str, fold: int, method: str, k: int, val_ratio: float, seed: int):
    if method == "kfold":
        assert k is not None
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for train, val in kf.split(files):
            folds.append({"train": list(files[train]), "val": list(files[val])})
        splits = SplitConfig(folds, fold, method, seed=seed, k=k, seed=seed)

    elif method == "simple":
        assert val_ratio is not None
        np.random.seed(seed)
        np.random.shuffle(files)  # inplace
        num_val = math.ceil(len(files) * val_ratio)
        if num_val < 10:
            logging.warning("The validation split is very small. Consider using a higher `p`.")

        folds = [{"train": list(files[train]), "val": list(files[val])}]
        splits = SplitConfig(folds, fold, method, val_ratio=val_ratio, seed=seed)
    else:
        raise ValueError("`method` is not a valid SplitMethod")

    save_pickle(splits, splits_path)
    return splits


def load_files(train_data_dir):
    files = subfiles(train_data_dir, join=False, suffix=".npy")
    if not files:
        files = subfiles(train_data_dir, join=False, suffix=".npz")
        if files:
            logging.warn("Only found compressed (.npz) files. This might increase runtime.")

    assert files, f"Couldn't find any .npy or .npz files in {train_data_dir}"
    return np.array(files)
