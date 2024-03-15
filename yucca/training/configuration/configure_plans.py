import yucca
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from dataclasses import dataclass
from typing import Union, Literal
from yucca.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.utils.dict import without_keys
from yucca.utils.files_and_folders import recursive_find_python_class
import logging


@dataclass
class PlanConfig:
    image_extension: str
    num_classes: int
    plans: dict
    task_type: str

    def lm_hparams(self, without: [] = []):
        hparams = {
            "image_extension": self.image_extension,
            "num_classes": self.num_classes,
            "plans": self.plans,
            "task_type": self.task_type,
        }
        return without_keys(hparams, without)


def get_plan_config(
    plans_path: str,
    stage: Literal["fit", "test", "predict"],
    ckpt_plans: Union[dict, None] = None,
):
    assert stage in ["fit", "test", "predict"], f"stage: {stage} is not supported"
    # First try to load torch checkpoints and extract plans and carry-over information from there.
    if stage == "fit":
        plans = load_plans(plans_path)
    if stage == "test":
        raise NotImplementedError
    if stage == "predict":
        # In this case we don't want to rely on plans being found in the preprocessed folder,
        # as it might not exist.
        assert ckpt_plans is not None
        plans = ckpt_plans

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
    keys_to_drop = ["original_sizes", "original_spacings", "new_sizes", "new_spacings"]
    logging.info("Loading plans.json")
    plan = load_json(plans_path)
    for key in keys_to_drop:
        if key in plan:
            plan.pop(key)
        elif key in plan["dataset_properties"]:
            plan["dataset_properties"].pop(key)
    return plan


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
        task_type = "self-supervised"
    else:
        task_type = "segmentation"
    return task_type
