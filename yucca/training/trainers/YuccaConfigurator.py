import torch
import yucca
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    load_json,
    load_pickle,
    isfile,
    isdir,
    save_pickle,
    subfiles,
    subdirs,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from sklearn.model_selection import KFold
from yucca.preprocessing.UnsupervisedPreprocessor import UnsupervisedPreprocessor
from yucca.preprocessing.ClassificationPreprocessor import ClassificationPreprocessor
from yucca.paths import yucca_models, yucca_preprocessed_data
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from yuccalib.utils.files_and_folders import WritePredictionFromLogits, recursive_find_python_class
from yuccalib.evaluation.loggers import YuccaLogger
from typing import Union, Literal


class YuccaConfigurator:
    """
    The YuccaConfigurator class is a configuration manager designed for the Yucca project.
    It is responsible for handling various configurations related to training, logging, model checkpoints, and data loading.
    This class streamlines the setup of essential components for training a neural network using PyTorch Lightning.

    task (str): The task or dataset name (e.g., "Task001_OASIS").

    continue_from_most_recent (bool, optional): Whether to continue training from the newest version. Default is True.
        - When this is True the Configurator will look for previous trainings and resume the latest version.
        - When this is False the Configurator will look for previous trainings and start a new training with a
        version one higher than the latest

    folds (str, optional): Fold identifier for cross-validation. Default is "0".

    max_vram (int, optional): Maximum VRAM (Video RAM) usage in gigabytes. Default is 12.

    manager_name (str, optional): Name of the manager associated with the pipeline. Default is "YuccaLightningManager".

    model_dimensions (str, optional): Model dimensionality ("2D" or "3D"). Default is "3D".

    model_name (str, optional): Model architecture name. Default is "UNet".

    planner (str, optional): Name of the planner associated with the dataset. Default is "YuccaPlanner".

    patch_size (tuple, optional): Patch size. Can be a tuple (2D), tuple (3D), string, or None. Default is None.
        - If a tuple is provided: the patch size will be set to that value.
        - If a string is provided:  "max" will set the patch size to the maximum input of the dataset
                                    "min" will set the patch size to the minimum input of the dataset
                                    "mean" will set the patch size to the mean input of the dataset
        - If None is provided: the patch size will be inferred.

    batch_size: Batch size. Can be either an integer or None. Default is None.
        - If an interger is provided: the batch size will be set to that value.
        - If None is provided: the batch size will be inferred.
    """

    def __init__(
        self,
        task: str,
        ckpt_path: str = None,
        continue_from_most_recent: bool = True,
        split_idx: int = 0,
        max_vram: int = 12,
        manager_name: str = "YuccaLightningManager",
        model_dimensions: str = "3D",
        model_name: str = "UNet",
        planner: str = "YuccaPlanner",
        patch_size: Union[tuple, Literal["max", "min", "mean", "tiny"]] = None,
        batch_size: int = None,
    ):
        self.ckpt_path = ckpt_path
        self.continue_from_most_recent = continue_from_most_recent
        self.split_idx = split_idx
        self.max_vram = max_vram
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.manager_name = manager_name
        self.planner = planner
        self.task = task
        self.patch_size = patch_size
        self.batch_size = batch_size

        # Attributes set upon calling
        self._plans = None
        self._version = None

        # Now run the setup
        self.setup_paths()
        self.setup_data_params()
        self.setup_aug_params()
        self.populate_lm_hparams()

    @property
    def plans(self):
        if self._plans is None:
            if self.ckpt_path is not None:
                print("Trying to find plans in specified ckpt")
                self._plans = torch.load(self.ckpt_path, map_location="cpu")["hyper_parameters"].get("config['plans']")
            elif (
                self.version is not None
                and self.continue_from_most_recent
                and isfile(join(self.version_dir, "checkpoints", "last.ckpt"))
            ):
                print("Trying to find plans in last ckpt")
                self._plans = torch.load(join(self.version_dir, "checkpoints", "last.ckpt"), map_location="cpu")[
                    "hyper_parameters"
                ].get("config['plans']")
            # If plans is still none the ckpt files were either empty/invalid or didn't exist and we create a new.
            if self._plans is None:
                print("Exhausted other options: loading plans.json and constructing parameters")
                self._plans = load_json(self.plans_path)
        return self._plans

    @property
    def version(self) -> Union[None, int]:
        if self._version:
            return self._version

        # If the dir doesn't exist we return version 0
        if not isdir(self.save_dir):
            self._version = 0
            return self._version

        # The dir exists. Check if any previous version exists in dir.
        previous_versions = subdirs(self.save_dir, join=False)

        # If no previous version exists we return version 0
        if not previous_versions:
            self._version = 0
            return self._version

        # If previous version(s) exists we can either (1) continue from the newest or
        # (2) create the next version
        if previous_versions:
            newest_version = max([int(i.split("_")[-1]) for i in previous_versions])
            if self.continue_from_most_recent:
                self._version = newest_version
            else:
                self._version = newest_version + 1
            return self._version

    def get_model_weights(self) -> {}:
        print(f"loading weights from {self.ckpt_path}")
        return torch.load(self.ckpt_path, map_location=torch.device("cpu"))["state_dict"]

    def setup_paths(self):
        self.train_data_dir = join(yucca_preprocessed_data, self.task, self.planner)
        self.save_dir = join(
            yucca_models,
            self.task,
            self.model_name + "__" + self.model_dimensions,
            self.manager_name + "__" + self.planner,
            f"fold_{self.split_idx}",
        )
        self.version_dir = join(self.save_dir, f"version_{self.version}")
        maybe_mkdir_p(self.version_dir)
        self.plans_path = join(yucca_preprocessed_data, self.task, self.planner, self.planner + "_plans.json")

    def setup_data_params(self):
        # (1) check if previous versions exist
        # (2) check if we want to continue from this
        # (3) check if hparams were created for the previous version

        self.num_classes = max(1, self.plans.get("num_classes") or len(self.plans["dataset_properties"]["classes"]))
        self.num_modalities = max(1, self.plans.get("num_modalities") or len(self.plans["dataset_properties"]["modalities"]))
        self.image_extension = (
            self.plans.get("image_extension") or self.plans["dataset_properties"].get("image_extension") or "nii.gz"
        )
        if self.plans.get("batch_size") and self.plans.get("patch_size"):
            # load batch and patch size from plan if they exist
            self.batch_size = self.plans.get("batch_size")
            self.patch_size = self.plans.get("patch_size")
        else:
            if not torch.cuda.is_available():
                # tiny patch and batch size for CPU training
                if self.batch_size is None:
                    self.batch_size = 2
                if self.patch_size is None or self.patch_size == "tiny":
                    self.patch_size = (32, 32) if self.model_dimensions == "2D" else (32, 32, 32)
            else:
                # We either have to infer patch size, batch size, both, or none, thus we have to check four cases
                # Case 1: infer patch size, infer batch size
                if self.patch_size is None and self.batch_size is None:
                    self.batch_size, self.patch_size = find_optimal_tensor_dims(
                        dimensionality=self.model_dimensions,
                        num_classes=self.num_classes,
                        modalities=self.num_modalities,
                        model_name=self.model_name,
                        max_patch_size=self.plans["new_mean_size"],
                        max_memory_usage_in_gb=self.max_vram,
                    )
                # Case 2: infer patch size, fixed batch size
                elif self.patch_size is None and self.batch_size is not None:
                    raise NotImplementedError  # Not yet supported in Yuccalib

                # Case 3: fixed patch size, infer batch size
                elif self.patch_size is not None and self.batch_size is None:
                    # Get patch size from dataset
                    if isinstance(self.patch_size, str):
                        if self.patch_size in ["max", "min", "mean"]:
                            self.patch_size = self.plans[f"new_{self.patch_size}_size"]
                        elif self.patch_size == "tiny":
                            self.patch_size = (32, 32, 32)

                        if self.model_dimensions == "2D" and len(self.patch_size) == 3:
                            # If we have now selected a 3D patch for a 2D model we skip the first dim
                            # as we will be extracting patches from that dimension.
                            self.patch_size = self.patch_size[1:]

                    self.batch_size, self.patch_size = find_optimal_tensor_dims(
                        fixed_patch_size=self.patch_size,
                        dimensionality=self.model_dimensions,
                        num_classes=self.num_classes,
                        modalities=self.num_modalities,
                        model_name=self.model_name,
                        max_patch_size=self.plans["new_mean_size"],
                        max_memory_usage_in_gb=self.max_vram,
                    )

                # Case 4: fixed patch size, fixed batch size
                elif self.patch_size is not None and self.batch_size is not None:
                    print("Using provided patch and batch sizes.")
                    # do nothing. Patch and batch size already set!

        assert isinstance(self.patch_size, tuple), self.patch_size
        assert isinstance(self.batch_size, int), self.batch_size
        assert self.batch_size > 0, self.batch_size
        assert (self.model_dimensions == "2D" and len(self.patch_size) == 2) or (
            self.model_dimensions == "3D" and len(self.patch_size) == 3
        ), (self.model_dimensions, len(self.patch_size))

        print(f"Using batch size: {self.batch_size} and patch size: {self.patch_size}")

    def setup_aug_params(self):
        preprocessor_class = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "preprocessing")],
            class_name=self.plans["preprocessor"],
            current_module="yucca.preprocessing",
        )
        assert (
            preprocessor_class
        ), f"{self.plans['preprocessor']} was found in plans, but no class with the corresponding name was found"
        self.augmentation_parameter_dict = {}
        if issubclass(preprocessor_class, ClassificationPreprocessor):
            self.augmentation_parameter_dict["skip_label"] = True
            self.task_type = "classification"
        elif issubclass(preprocessor_class, UnsupervisedPreprocessor):
            self.augmentation_parameter_dict["skip_label"] = True
            self.augmentation_parameter_dict["copy_image_to_label"] = True
            # This should be uncommented when masking is properly implemented
            # self.augmentation_parameter_dict["mask_image_for_reconstruction"] = True
            self.task_type = "unsupervised"
        else:
            self.task_type = "segmentation"

    def populate_lm_hparams(self):
        # Here we control which variables go into the hparam dict that we pass to the LightningModule
        # This way we can make sure it won't be given args that could potentially break it (such as classes that might be changed)
        # or flood it with useless information.
        self.lm_hparams = {
            "aug_params": self.augmentation_parameter_dict,
            "batch_size": self.batch_size,
            "ckpt_path": self.ckpt_path,
            "continue_from_most_recent": self.continue_from_most_recent,
            "image_extension": self.image_extension,
            "manager_name": self.manager_name,
            "max_vram": self.max_vram,
            "model_dimensions": self.model_dimensions,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_modalities": self.num_modalities,
            "version_dir": self.version_dir,
            "patch_size": self.patch_size,
            "planner": self.planner,
            "plans_path": self.plans_path,
            "plans": self.plans,
            "task_type": self.task_type,
            "save_dir": self.save_dir,
            "task": self.task,
            "train_data_dir": self.train_data_dir,
        }


if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger, CSVLogger

    x = YuccaConfigurator(
        task="Task001_OASIS", model_name="TinyUNet", model_dimensions="2D", manager_name="YuccaLightningManager"
    )
    # x.val_split
