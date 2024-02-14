from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p
from dataclasses import dataclass
from typing import Union
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.training.configuration.configure_task import TaskConfig


@dataclass
class PathConfig:
    plans_path: str
    save_dir: str
    task_dir: str
    train_data_dir: str
    version_dir: str
    version: int

    def lm_hparams(self):
        return {
            "plans_path": self.plans_path,
            "save_dir": self.save_dir,
            "train_data_dir": self.train_data_dir,
            "version_dir": self.version_dir,
            "version": self.version,
        }


def get_path_config(task_config: TaskConfig):
    task_dir = join(yucca_preprocessed_data, task_config.task)
    train_data_dir = join(task_dir, task_config.planner_name)
    save_dir = join(
        yucca_models,
        task_config.task,
        task_config.model_name + "__" + task_config.model_dimensions,
        task_config.manager_name + "__" + task_config.planner_name,
        task_config.experiment,
        f"{task_config.split_method}_{task_config.split_param}_fold_{task_config.split_idx}",
    )

    version = detect_version(save_dir, task_config.continue_from_most_recent)
    version_dir = join(save_dir, f"version_{version}")
    maybe_mkdir_p(version_dir)
    plans_path = join(task_dir, task_config.planner_name, task_config.planner_name + "_plans.json")

    return PathConfig(
        plans_path=plans_path,
        save_dir=save_dir,
        task_dir=task_dir,
        train_data_dir=train_data_dir,
        version_dir=version_dir,
        version=version,
    )


def detect_version(save_dir, continue_from_most_recent) -> Union[None, int]:
    # If the dir doesn't exist we return version 0
    if not isdir(save_dir):
        return 0

    # The dir exists. Check if any previous version exists in dir.
    previous_versions = subdirs(save_dir, join=False)
    # If no previous version exists we return version 0
    if not previous_versions:
        return 0

    # If previous version(s) exists we can either (1) continue from the newest or
    # (2) create the next version
    if previous_versions:
        newest_version = max([int(i.split("_")[-1]) for i in previous_versions])
        if continue_from_most_recent:
            return newest_version
        else:
            return newest_version + 1
