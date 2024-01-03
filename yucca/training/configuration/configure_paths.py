import yucca
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p, isfile, load_json
from dataclasses import dataclass
from typing import Union
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.utils.files_and_folders import recursive_find_python_class
from yucca.training.configuration.configure_task import TaskConfig


@dataclass
class PathConfig:
    plans_path: str
    save_dir: str
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
    save_dir, train_data_dir, version_dir, plans_path, version = setup_paths_and_version(
        task_config.continue_from_most_recent,
        task_config.manager_name,
        task_config.model_dimensions,
        task_config.model_name,
        task_config.split_idx,
        task_config.task,
        task_config.planner_name,
        task_config.experiment,
    )

    return PathConfig(
        plans_path=plans_path,
        save_dir=save_dir,
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


def setup_paths_and_version(
    continue_from_most_recent, manager_name, model_dimensions, model_name, split_idx, task, planner, experiment
):
    train_data_dir = join(yucca_preprocessed_data, task, planner)
    save_dir = join(
        yucca_models,
        task,
        model_name + "__" + model_dimensions,
        manager_name + "__" + planner,
        experiment,
        f"fold_{split_idx}",
    )

    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = join(save_dir, f"version_{version}")
    maybe_mkdir_p(version_dir)
    plans_path = join(yucca_preprocessed_data, task, planner, planner + "_plans.json")
    return save_dir, train_data_dir, version_dir, plans_path, version
