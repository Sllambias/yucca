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
)
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from sklearn.model_selection import KFold
from yucca.paths import yucca_models, yucca_preprocessed
from yuccalib.image_processing.matrix_ops import get_max_rotated_size
from yuccalib.network_architectures.utils.model_memory_estimation import (
    find_optimal_tensor_dims,
)
from yuccalib.utils.files_and_folders import WriteSegFromLogits
from yuccalib.evaluation.loggers import TXTLogger


class YuccaConfigurator:
    def __init__(
        self,
        folds: str = "0",
        tiny_patch: bool = False,
        max_vram: int = 12,
        manager_name: str = None,
        model_dimensions: str = "3D",
        model_name: str = "UNet",
        planner: str = "YuccaPlanner",
        segmentation_output_dir: str = "./",
        task: str = None,
    ):
        self.folds = folds
        self.tiny_patch = tiny_patch
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.manager_name = manager_name
        self.segmentation_output_dir = segmentation_output_dir
        self.planner = planner
        self.task = task
        self.max_vram = max_vram
        self.run_setup()

    def run_setup(self):
        self.setup_paths_and_plans()
        self.setup_splits()
        self.setup_loggers()
        self.setup_callbacks()
        self.setup_train_params()

    def setup_paths_and_plans(self):
        self.train_data_dir = join(yucca_preprocessed, self.task, self.planner)

        self.outpath = join(
            yucca_models,
            self.task,
            self.model_name + "__" + self.model_dimensions,
            self.planner,
            self.manager_name,
            f"fold_{self.folds}",
        )

        maybe_mkdir_p(self.outpath)

        self.plans_path = join(yucca_preprocessed, self.task, self.planner, self.planner + "_plans.json")

        self.plans = load_json(self.plans_path)

    def setup_splits(self):
        # Load splits file or create it if not found (see: "split_data").
        splits_file = join(yucca_preprocessed, self.task, "splits.pkl")
        if not isfile(splits_file):
            self.split_data(splits_file)

        splits_file = load_pickle(join(yucca_preprocessed, self.task, "splits.pkl"))
        self.train_split = splits_file[int(self.folds)]["train"]
        self.val_split = splits_file[int(self.folds)]["val"]

    def setup_loggers(self):
        csvlogger = CSVLogger(save_dir=self.outpath, name=None)

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
        self.loggers = [csvlogger, wandb_logger, txtlogger]

    def setup_callbacks(self):
<<<<<<< Updated upstream
        pred_writer = WriteSegFromLogits(output_dir=self.segmentation_output_dir, write_interval="batch")
        self.callbacks = [pred_writer]
=======
        best_ckpt = ModelCheckpoint(monitor="val_dice", save_top_k=1, filename="model_best")
        interval_ckpt = ModelCheckpoint(every_n_epochs=250, filename="model_epoch_{epoch}")
        pred_writer = WriteSegFromLogits(output_dir=self.segmentation_output_dir, write_interval="batch")

        self.callbacks = [best_ckpt, interval_ckpt, pred_writer]
>>>>>>> Stashed changes

    def setup_train_params(self):
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

    csvlogger = CSVLogger(
        save_dir="/home/zcr545/YuccaData/yucca_models/Task001_OASIS/UNet__3D/YuccaLightningManager__YuccaPlanner/0", name=None
    )

# %%
