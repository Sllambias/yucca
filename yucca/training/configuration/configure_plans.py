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
class PlanConfig:
    image_extension: str
    num_classes: int
    plans: dict
    task_type: str

    def lm_hparams(self):
        return {
            "image_extension": self.image_extension,
            "num_classes": self.num_classes,
            "plans": self.plans,
            "task_type": self.task_type,
        }


def get_plan_config(ckpt_path: str, continue_from_most_recent: bool, plans_path: str, version: int, version_dir: str):
    plans = setup_plans(ckpt_path, continue_from_most_recent, plans_path, version, version_dir)
    task_type = setup_task_type(plans)
    num_classes = max(1, plans.get("num_classes") or len(plans["dataset_properties"]["classes"]))
    image_extension = plans.get("image_extension") or plans["dataset_properties"].get("image_extension") or "nii.gz"

    return PlanConfig(
        image_extension=image_extension,
        num_classes=num_classes,
        plans=plans,
        task_type=task_type,
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


def setup_task_type(plans):
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
    return task_type
