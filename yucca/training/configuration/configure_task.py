from dataclasses import dataclass
from typing import Optional, Union
from enum import StrEnum


class SplitMethods(StrEnum):
    kfold = "kfold"
    simple_train_val_split = "simple_train_val_split"


@dataclass
class TaskConfig:
    continue_from_most_recent: bool
    manager_name: str
    model_dimensions: str
    model_name: str
    patch_based_training: bool
    planner_name: str
    split_idx: int
    task: str
    experiment: str
    split_method: SplitMethods
    split_param: Union[float, int]

    def lm_hparams(self):
        return {
            "continue_from_most_recent": self.continue_from_most_recent,
            "manager_name": self.manager_name,
            "model_dimensions": self.model_dimensions,
            "model_name": self.model_name,
            "patch_based_training": self.patch_based_training,
            "planner_name": self.planner_name,
            "task": self.task,
            "split_idx": self.split_idx,
            "experiment": self.experiment,
        }


def get_task_config(
    task: str,
    continue_from_most_recent: bool = True,
    manager_name: str = "YuccaManager",
    model_dimensions: str = "3D",
    model_name: str = "UNet",
    planner_name: str = "YuccaPlanner",
    patch_based_training: bool = True,
    experiment: str = "default",
    split_idx: int = 0,
    split_data_kfold: Optional[int] = 5,
    split_data_ratio: Optional[float] = None,
):
    assert model_dimensions is not None

    split_method, split_param = split_method_and_param(split_data_kfold, split_data_ratio)

    return TaskConfig(
        task=task,
        continue_from_most_recent=continue_from_most_recent,
        manager_name=manager_name,
        model_dimensions=model_dimensions,
        model_name=model_name,
        patch_based_training=patch_based_training,
        planner_name=planner_name,
        experiment=experiment,
        split_idx=split_idx,
        split_method=split_method,
        split_param=split_param,
    )


def split_method_and_param(split_data_kfold, split_data_ratio):
    """
    Note:
        You can only provide one of `k` or `split_data_ratio`.
        - If `k` is provided we will split with `k-fold`.
        - If `split_data_ratio` is provided it determines the fraction of items used for the val split.
    """

    assert (split_data_kfold is not None and split_data_ratio is None) or (
        split_data_kfold is None and split_data_ratio is not None
    ), "You must provide excatly one of either `split_data_kfold  or `split_data_ratio`."
    if split_data_ratio is not None:
        assert (
            0 < split_data_ratio < 1
        ), "`split_data_ratio` must be a number between 0 and 1 and determines the fraction of items used for the val split"
    if split_data_kfold is not None:
        assert split_data_kfold > 0
        assert isinstance(split_data_kfold, int), "`split_data_kfold  should be an integer"

    if split_data_kfold is not None:
        return str(SplitMethods.kfold), split_data_kfold
    else:  # split_data_ratio is not None
        return str(SplitMethods.simple_train_val_split), split_data_ratio
