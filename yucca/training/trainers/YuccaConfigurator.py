# %%
import torch
import numpy as np
from time import localtime, strftime
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
from yucca.paths import yucca_models, yucca_preprocessed
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from yuccalib.utils.files_and_folders import WriteSegFromLogits, load_yaml
from yuccalib.evaluation.loggers import YuccaLogger
from typing import Union


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

    disable_logging (bool, optional): Whether to disable logging. Default is False.
        - This disables both the training log file and WandB logging. hparams.yaml will still be saved.

    folds (str, optional): Fold identifier for cross-validation. Default is "0".

    max_vram (int, optional): Maximum VRAM (Video RAM) usage in gigabytes. Default is 12.

    manager_name (str, optional): Name of the manager associated with the pipeline. Default is "YuccaLightningManager".

    model_dimensions (str, optional): Model dimensionality ("2D" or "3D"). Default is "3D".

    model_name (str, optional): Model architecture name. Default is "UNet".

    planner (str, optional): Name of the planner associated with the dataset. Default is "YuccaPlanner".

    segmentation_output_dir (str, optional): Output directory for segmentation results. Default is "./".
        - Only used during inference.

    save_softmax (bool, optional): Whether to save softmax predictions during inference. Default is False.
        - Only used during inference. Used to save the softmax predictions combined by model ensembles.

    tiny_patch (bool, optional): Whether to use a tiny patch size. Default is False.
        - Mainly useful for debugging and/or running on CPU.
    """

    def __init__(
        self,
        task: str,
        continue_from_most_recent: bool = True,
        disable_logging: bool = False,
        folds: str = "0",
        max_vram: int = 12,
        manager_name: str = "YuccaLightningManager",
        model_dimensions: str = "3D",
        model_name: str = "UNet",
        planner: str = "YuccaPlanner",
        profile: bool = False,
        segmentation_output_dir: str = "./",
        save_softmax: bool = False,
        tiny_patch: bool = False,
    ):
        self.continue_from_most_recent = continue_from_most_recent
        self.folds = folds
        self.disable_logging = disable_logging
        self.max_vram = max_vram
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.manager_name = manager_name
        self.save_softmax = save_softmax
        self.segmentation_output_dir = segmentation_output_dir
        self.planner = planner
        self.profile = profile
        self.task = task
        self.tiny_patch = tiny_patch

        # Attributes set upon calling
        self._plans = None
        self._train_split = None
        self._val_split = None
        self._version = None
        self._profiler = None

        # Now run the setup
        self.setup_paths()
        self.setup_data_params()
        self.setup_callbacks()
        self.setup_loggers()

    @property
    def plans(self):
        if self._plans is None:
            self._plans = load_json(self.plans_path)
        return self._plans

    @property
    def profiler(self):
        if self.profile and self._profiler is None:
            self._profiler = SimpleProfiler(dirpath=join(self.outpath, f"version_{self.version}"), filename="simple_profile")
        return self._profiler

    @property
    def train_split(self):
        # We do not want to make "self.load_splits" a part of the default pipeline, since many
        # test sets will not have a training set and thus no splits. E.g. training on
        # DatasetA with a training set and running inference on DatasetB with no training set.
        if self._train_split is None:
            self.load_splits()
        return self._train_split

    @property
    def val_split(self):
        if self._val_split is None:
            self.load_splits()
        return self._val_split

    @property
    def version(self) -> Union[None, int]:
        if self._version:
            return self._version

        # If the dir doesn't exist we return version 0
        if not isdir(self.outpath):
            self._version = 0
            return self._version

        # The dir exists. Check if any previous version exists in dir.
        previous_versions = subdirs(self.outpath, join=False)

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

    def setup_loggers(self):
        # The CSVLogger is the barebones logger needed to save hparams.yaml and set the proper
        # outpath that will be expected by the pipeline for continued training etc.
        # It should generally never be disabled.
        self.loggers = []
        # self.loggers.append(CSVLogger(save_dir=self.outpath, name=None, version=self.version))
        self.loggers.append(
            YuccaLogger(
                disable_logging=self.disable_logging,
                save_dir=self.outpath,
                name=None,
                version=self.version,
                steps_per_epoch=250,
            )
        )
        if not self.disable_logging:
            self.loggers.append(
                WandbLogger(
                    name=f"version_{self.version}",
                    save_dir=join(self.outpath, f"version_{self.version}"),
                    version=self.version,
                    project="Yucca",
                    group=self.task,
                    log_model="all",
                )
            )

    def setup_callbacks(self):
        best_ckpt = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best", enable_version_counter=True)
        interval_ckpt = ModelCheckpoint(every_n_epochs=250, filename="{epoch}", enable_version_counter=False)
        latest_ckpt = ModelCheckpoint(
            every_n_epochs=15,
            save_top_k=1,
            filename="last",
            enable_version_counter=False,
        )
        pred_writer = WriteSegFromLogits(
            output_dir=self.segmentation_output_dir, save_softmax=self.save_softmax, write_interval="batch"
        )
        self.callbacks = [best_ckpt, interval_ckpt, latest_ckpt, pred_writer]

    def setup_paths(self):
        self.train_data_dir = join(yucca_preprocessed, self.task, self.planner)
        self.outpath = join(
            yucca_models,
            self.task,
            self.model_name + "__" + self.model_dimensions,
            self.manager_name + "__" + self.planner,
            f"fold_{self.folds}",
        )
        maybe_mkdir_p(self.outpath)
        self.plans_path = join(yucca_preprocessed, self.task, self.planner, self.planner + "_plans.json")

    def setup_data_params(self):
        # (1) check if previous versions exist
        # (2) check if we want to continue from this
        # (3) check if hparams were created for the previous version
        if (
            self.version is not None
            and self.continue_from_most_recent
            and isfile(join(self.outpath, f"version_{self.version}", "hparams.yaml"))
        ):
            print("Found hparams.yaml, loading parameters")
            hparams = load_yaml(join(self.outpath, f"version_{self.version}", "hparams.yaml"))
            self.num_classes = int(hparams["configurator"]["num_classes"])
            self.num_modalities = int(hparams["configurator"]["num_modalities"])
            self.batch_size = int(hparams["configurator"]["batch_size"])
            self.patch_size = [int(p) for p in hparams["configurator"]["patch_size"]]
        else:
            print("Constructing new parameters")
            self.num_classes = len(self.plans["dataset_properties"]["classes"])
            self.num_modalities = len(self.plans["dataset_properties"]["modalities"])
            if self.tiny_patch or not torch.cuda.is_available():
                self.batch_size = 2
                self.patch_size = (32, 32) if self.model_dimensions == "2D" else (32, 32, 32)
            else:
                self.batch_size, self.patch_size = find_optimal_tensor_dims(
                    dimensionality=self.model_dimensions,
                    num_classes=self.num_classes,
                    modalities=self.num_modalities,
                    model_name=self.model_name,
                    max_patch_size=self.plans["new_mean_size"],
                    max_memory_usage_in_gb=self.max_vram,
                )
        print(f"Using batch size: {self.batch_size} and patch size: {self.patch_size}")

    def load_splits(self):
        # Load splits file or create it if not found (see: "split_data").
        splits_file = join(yucca_preprocessed, self.task, "splits.pkl")
        if not isfile(splits_file):
            self.split_data(splits_file)

        splits_file = load_pickle(join(yucca_preprocessed, self.task, "splits.pkl"))
        self._train_split = splits_file[int(self.folds)]["train"]
        self._val_split = splits_file[int(self.folds)]["val"]

    def split_data(self, splits_file):
        splits = []

        files = subfiles(self.train_data_dir, join=False, suffix=".npy")
        if not files:
            files = subfiles(self.train_data_dir, join=False, suffix=".npz")
            if files:
                self.log(
                    "Only found compressed (.npz) files. This might increase runtime.",
                    time=False,
                )

        assert files, f"Couldn't find any .npy or .npz files in {self.train_data_dir}"

        files = np.array(files)
        # We set this seed manually as multiple trainers might use this split,
        # And we may not know which individual seed dictated the data splits
        # Therefore for reproducibility this is fixed.

        kf = KFold(n_splits=5, shuffle=True, random_state=52189)
        for train, val in kf.split(files):
            splits.append({"train": list(files[train]), "val": list(files[val])})

        save_pickle(splits, splits_file)


# %%
if __name__ == "__main__":
    from pytorch_lightning.loggers import WandbLogger, CSVLogger

    x = YuccaConfigurator(
        task="Task001_OASIS", model_name="TinyUNet", model_dimensions="2D", manager_name="YuccaLightningManager"
    )
    # x.val_split
# %%
