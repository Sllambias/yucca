# %%
import yucca
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p, isfile, load_json
from dataclasses import dataclass
from typing import Union
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.utils.files_and_folders import recursive_find_python_class


@dataclass
class PathAndPlanConfig:
    continue_from_most_recent: bool
    ckpt_path: str
    image_extension: str
    manager_name: str
    model_dimensions: str
    model_name: str
    num_classes: int
    planner: str
    plans: dict
    plans_path: str
    save_dir: str
    split_idx: int
    task: str
    task_type: str
    train_data_dir: str
    version_dir: str
    version: int

    def lm_hparams(self):
        return {
            "continue_from_most_recent": self.continue_from_most_recent,
            "ckpt_path": self.ckpt_path,
            "image_extension": self.image_extension,
            "manager_name": self.manager_name,
            "model_dimensions": self.model_dimensions,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "planner": self.planner,
            "plans": self.plans,
            "plans_path": self.plans_path,
            "save_dir": self.save_dir,
            "split_idx": self.split_idx,
            "task": self.task,
            "task_type": self.task_type,
            "train_data_dir": self.train_data_dir,
            "version_dir": self.version_dir,
            "version": self.version,
        }


def get_path_and_plan_config(
    task: str,
    ckpt_path: str = None,
    continue_from_most_recent: bool = True,
    manager_name: str = "YuccaLightningManager",
    model_dimensions: str = "3D",
    model_name: str = "UNet",
    planner: str = "YuccaPlanner",
    split_idx: int = 0,
):
    save_dir, train_data_dir, version_dir, plans_path, version = setup_paths_and_version(
        continue_from_most_recent, manager_name, model_dimensions, model_name, split_idx, task, planner
    )
    plans = setup_plans(ckpt_path, continue_from_most_recent, plans_path, version, version_dir)
    num_classes, task_type = setup_classes_and_task_type(plans)
    image_extension = plans.get("image_extension") or plans["dataset_properties"].get("image_extension") or "nii.gz"
    return PathAndPlanConfig(
        continue_from_most_recent=continue_from_most_recent,
        ckpt_path=ckpt_path,
        image_extension=image_extension,
        manager_name=manager_name,
        model_dimensions=model_dimensions,
        model_name=model_name,
        num_classes=num_classes,
        planner=planner,
        plans=plans,
        plans_path=plans_path,
        save_dir=save_dir,
        split_idx=split_idx,
        task=task,
        task_type=task_type,
        train_data_dir=train_data_dir,
        version_dir=version_dir,
        version=version,
    )


def setup_plans(ckpt_path, continue_from_most_recent, plans_path, version, version_dir):
    if ckpt_path is not None:
        print("Trying to find plans in specified ckpt")
        return torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]["config"]["plans"]
    elif version is not None and continue_from_most_recent and isfile(join(version_dir, "checkpoints", "last.ckpt")):
        print("Trying to find plans in last ckpt")
        return torch.load(join(version_dir, "checkpoints", "last.ckpt"), map_location="cpu")["hyper_parameters"]["config"][
            "plans"
        ]
    # If plans is still none the ckpt files were either empty/invalid or didn't exist and we create a new.
    print("Exhausted other options: loading plans.json and constructing parameters")
    return load_json(plans_path)


#
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


def setup_paths_and_version(continue_from_most_recent, manager_name, model_dimensions, model_name, split_idx, task, planner):
    train_data_dir = join(yucca_preprocessed_data, task, planner)
    save_dir = join(
        yucca_models,
        task,
        model_name + "__" + model_dimensions,
        manager_name + "__" + planner,
        f"fold_{split_idx}",
    )
    version = detect_version(save_dir, continue_from_most_recent)
    version_dir = join(save_dir, f"version_{version}")
    maybe_mkdir_p(version_dir)
    plans_path = join(yucca_preprocessed_data, task, planner, planner + "_plans.json")
    return save_dir, train_data_dir, version_dir, plans_path, version


def setup_classes_and_task_type(plans):
    num_classes = max(1, plans.get("num_classes") or len(plans["dataset_properties"]["classes"]))

    preprocessor_class = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "preprocessing")],
        class_name=plans["preprocessor"],
        current_module="yucca.preprocessing",
    )
    assert (
        preprocessor_class
    ), f"{plans['preprocessor']} was found in plans, but no class with the corresponding name was found"
    if issubclass(preprocessor_class, ClassificationPreprocessor):
        task_type = "classification"
    elif issubclass(preprocessor_class, UnsupervisedPreprocessor):
        task_type = "unsupervised"
    else:
        task_type = "segmentation"
    return num_classes, task_type


x = get_path_and_plan_config(task="Task001_OASIS")

# %%
