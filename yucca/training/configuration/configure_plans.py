import yucca
import torch
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subdirs, maybe_mkdir_p, isfile, load_json
from dataclasses import dataclass
from typing import Union, stage
from yucca.paths import yucca_models, yucca_preprocessed_data
from yucca.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.training.configuration.configure_paths import PathConfig
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


def get_plan_config(plans_path: str, stage: Literal["fit", "test", "predict"], ckpt_plans: Union[dict, None] = None, continue_from_most_recent: bool):
    # First try to load torch checkpoints and extract plans and carry-over information from there.
    if stage == "fit":
        plans = load_plans(plans_path)
    elif stage == "test":
        raise NotImplementedError
    elif stage == "predict":
        # In this case we don't want to rely on plans being found in the preprocessed folder,
        # as it might not exist.
        assert ckpt_plans is not None
        plans = ckpt_plans
    else:
        raise NotImplementedError(f'Stage: {stage} is not supported')
    
    task_type = setup_task_type(plans)
    num_classes = max(1, plans.get("num_classes") or len(plans["dataset_properties"]["classes"]))
    image_extension = plans.get("image_extension") or plans["dataset_properties"].get("image_extension") or "nii.gz"

    return PlanConfig(
        image_extension=image_extension,
        num_classes=num_classes,
        plans=plans,
        task_type=task_type,
    )


def load_plans(plans_path):
    # If plans is still none the ckpt files were either empty/invalid or didn't exist and we load the plans
    # from the preprocessed folder.
    print("Exhausted other options: loading plans.json")
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
