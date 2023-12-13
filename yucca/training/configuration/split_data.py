from dataclasses import dataclass
import logging
import math
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, isfile, save_pickle, load_pickle, load_files
from enum import StrEnum, auto

from sklearn.model_selection import KFold
from yucca.paths import yucca_preprocessed_data

# We set this seed manually as multiple trainers might use this split,
# And we may not know which individual seed dictated the data splits
# Therefore for reproducibility this is fixed.
DEFAULT_SEED = 52189

class SplitMethod(StrEnum):
    KFOLD = auto()
    SIMPLE = auto()

@dataclass
class Splits:
    folds: list[dict]
    method: SplitMethod
    seed: int
    k: int = None
    p: float = None

    def train(self, fold):
        return self.folds[fold]["train"]

    def val(self, fold):
        return self.folds[fold]["val"]


def split_data(train_data_dir: str, task: str, method: SplitMethod, k: int = 5, p: float = 0.01, seed: int = DEFAULT_SEED):
    """
    If method is `SplitMethod.KFOLD` then a k argument must be provided
    If method is `SplitMethod.SIMPLE` a p argument must be provided
    """
    splits_path = join(yucca_preprocessed_data, task, "splits.pkl")
    if isfile(splits_path):
        splits = load_pickle(splits_path)
        if isinstance(splits, Splits):
            return splits

    files = load_files(train_data_dir)
    return perform_split(files, splits_path, method, k, p, seed)


def perform_split(files: list[str], splits_path: str, method: SplitMethod, k: int, p: float, seed: int):
    if method == SplitMethod.KFOLD:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        folds = []
        for train, val in kf.split(files):
            folds.append({"train": list(files[train]), "val": list(files[val])}
        splits = Splits(folds, method, seed=seed, k=k, seed=seed)

    elif method == SplitMethod.SIMPLE:
        np.random.seed(seed)
        np.random.shuffle(files)  # inplace
        num_val = math.ceil(len(files) * p)
        if num_val < 10:
            logging.warning("The validation split is very small. Consider using a higher `p`.")

        folds = [{"train": list(files[train]), "val": list(files[val])}]
        splits = Splits(folds, method, p=p, seed=seed)
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
