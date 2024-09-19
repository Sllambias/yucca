import yucca
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from dataclasses import dataclass
from typing import Union, Literal
from yucca.pipeline.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.pipeline.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.functional.utils.dict import without_keys
from yucca.functional.utils.files_and_folders import recursive_find_python_class
from yucca.functional.utils.loading import load_yaml
import logging


@dataclass
class PlanConfig:
    allow_missing_modalities: bool
    image_extension: str
    num_classes: int
    plans: dict
    task_type: str
    use_label_regions: bool = False
    labels: dict = None
    regions: dict = None

    if use_label_regions:
        assert labels is not None
        assert regions is not None

    def lm_hparams(self, without: [] = []):
        hparams = {
            "allow_missing_modalities": self.allow_missing_modalities,
            "image_extension": self.image_extension,
            "num_classes": self.num_classes,
            "plans": self.plans,
            "task_type": self.task_type,
            "labels": self.labels,
            "regions": self.regions,
            "use_label_regions": self.use_label_regions,
        }
        return without_keys(hparams, without)


def get_plan_config(
    plans_path: str,
    stage: Literal["fit", "test", "predict"],
    ckpt_plans: Union[dict, None] = None,
    use_label_regions: bool = False,
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
        if isfile(plans_path):
            plans = load_yaml(plans_path)["config"]["plans"]
        else:
            assert ckpt_plans is not None
            plans = ckpt_plans

    regions = None
    labels = None

    task_type = setup_task_type(plans)
    if task_type == "self-supervised":
        num_classes = max(1, len(plans["dataset_properties"]["modalities"]))
    elif use_label_regions:
        labels = plans["dataset_properties"]["labels"]
        regions = plans["dataset_properties"]["regions"]
        num_classes = len(regions)
    else:
        labels = plans["dataset_properties"].get("labels") or plans["dataset_properties"].get("classes")
        num_classes = max(1, len(plans["dataset_properties"]["classes"]))
    image_extension = plans.get("image_extension") or plans["dataset_properties"].get("image_extension") or "nii.gz"
    allow_missing_modalities = plans.get("allow_missing_modalities") or False

    return PlanConfig(
        allow_missing_modalities=allow_missing_modalities,
        image_extension=image_extension,
        num_classes=num_classes,
        plans=plans,
        task_type=task_type,
        use_label_regions=use_label_regions,
        labels=labels,
        regions=regions,
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
    if plans.get("task_type"):
        task_type = plans.get("task_type")
        return task_type

    # If key is not present in plan then we try to infer the task_type from the Type of Preprocessor
    preprocessor_class = recursive_find_python_class(
        folder=[join(yucca.__path__[0], "pipeline", "preprocessing")],
        class_name=plans["preprocessor"],
        current_module="yucca.pipeline.preprocessing",
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
