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
    save_pickle,
    subfiles,
    subdirs,
)
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from sklearn.model_selection import KFold
from yucca.paths import yucca_models, yucca_preprocessed
from yuccalib.image_processing.matrix_ops import get_max_rotated_size
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from yuccalib.utils.files_and_folders import WriteSegFromLogits, load_yaml
from yuccalib.evaluation.loggers import TXTLogger
from typing import Union


class YuccaConfigurator:
    def __init__(
        self,
        continue_from_newest_version: bool = True,
        folds: str = "0",
        logging: bool = True,
        max_vram: int = 12,
        manager_name: str = None,
        model_dimensions: str = "3D",
        model_name: str = "UNet",
        planner: str = "YuccaPlanner",
        segmentation_output_dir: str = "./",
        task: str = None,
        tiny_patch: bool = False,
    ):
        self.continue_from_newest_version = continue_from_newest_version
        self.folds = folds
        self.logging = logging
        self.max_vram = max_vram
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.manager_name = manager_name
        self.segmentation_output_dir = segmentation_output_dir
        self.planner = planner
        self.task = task
        self.tiny_patch = tiny_patch

        # Attributes set upon calling
        self._train_split = None
        self._val_split = None
        self._version = None

        # Now run the setup
        self.setup_paths_and_plans()
        self.setup_train_params()

    def setup_paths_and_plans(self):
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
        self.plans = load_json(self.plans_path)

    @property
    def val_split(self):
        if not self._val_split:
            self.setup_splits()
        return self._val_split

    @property
    def train_split(self):
        if not self._train_split:
            self.setup_splits()
        return self._train_split

    @property
    def version(self) -> Union[None, int]:
        # Here we check if we want to continue from the newest version
        # And if it exists. If so the number of the newest version is returned
        # Otherwise None is returned.
        # If None is returned we create a new version.
        if self.continue_from_newest_version:
            previous_versions = subdirs(self.outpath, join=False)
            if previous_versions:
                self._version = int(max([i.split("_")[-1] for i in previous_versions]))
        return self._version

    @property
    def loggers(self):
        if not self.logging:
            return []
        csvlogger = CSVLogger(save_dir=self.outpath, name=None, version=self.version)

        wandb_logger = WandbLogger(
            name=f"version_{csvlogger.version}",
            save_dir=join(self.outpath, f"version_{csvlogger.version}"),
            project="Yucca",
            group=self.task,
            log_model="all",
        )

        txtlogger = TXTLogger(
            save_dir=self.outpath,
            name=f"version_{csvlogger.version}",
            steps_per_epoch=250,
        )
        return [csvlogger, wandb_logger, txtlogger]

    @property
    def callbacks(self):
        best_ckpt = ModelCheckpoint(monitor="val_dice", save_top_k=1, filename="model_best")
        interval_ckpt = ModelCheckpoint(every_n_epochs=250, filename="model_epoch_{epoch}")
        pred_writer = WriteSegFromLogits(output_dir=self.segmentation_output_dir, write_interval="batch")
        return [best_ckpt, interval_ckpt, pred_writer]

    def setup_splits(self):
        # Load splits file or create it if not found (see: "split_data").
        splits_file = join(yucca_preprocessed, self.task, "splits.pkl")
        if not isfile(splits_file):
            self.split_data(splits_file)

        splits_file = load_pickle(join(yucca_preprocessed, self.task, "splits.pkl"))
        self._train_split = splits_file[int(self.folds)]["train"]
        self._val_split = splits_file[int(self.folds)]["val"]

    def setup_train_params(self):
        # First check if we want to continue from newest version and if any versions exist
        print(self.version)
        print(isfile(join(self.outpath, f"version_{self.version}", "hparams.yaml")))

        if (
            self.version is not None
            and self.continue_from_newest_version
            and isfile(join(self.outpath, f"version_{self.version}", "hparams.yaml"))
        ):
            print("Loading hparams.yaml")
            hparams = load_yaml(join(self.outpath, f"version_{self.version}", "hparams.yaml"))
            self.num_classes = hparams["configurator"]["num_classes"]
            self.num_modalities = hparams["num_modalities"]
            self.batch_size = hparams["batch_size"]
            self.patch_size = hparams["patch_size"]
            self.initial_patch_size = hparams["initial_patch_size"]
        else:
            print("constructing new params")
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
            self.initial_patch_size = get_max_rotated_size(self.patch_size)

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
