from dataclasses import dataclass
from typing import Union


@dataclass
class TaskConfig:
    continue_from_most_recent: bool
    experiment: str
    manager_name: str
    model_dimensions: str
    model_name: str
    patch_based_training: bool
    planner_name: str
    split_idx: int
    split_method: str
    split_param: Union[str, float, int]
    task: str

    def lm_hparams(self):
        return {
            "continue_from_most_recent": self.continue_from_most_recent,
            "experiment": self.experiment,
            "manager_name": self.manager_name,
            "model_dimensions": self.model_dimensions,
            "model_name": self.model_name,
            "patch_based_training": self.patch_based_training,
            "planner_name": self.planner_name,
            "task": self.task,
            "split_idx": self.split_idx,
            "split_method": self.split_method,
            "split_param": self.split_param,
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
    split_data_method: str = "kfold",
    split_data_param: Union[str, float, int] = 5,
):
    assert model_dimensions is not None

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
        split_method=split_data_method,
        split_param=split_data_param,
    )
