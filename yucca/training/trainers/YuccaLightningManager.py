# %%
import lightning as L
from typing import Literal
from yucca.training.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.trainers.YuccaConfigurator import YuccaConfigurator
from yucca.training.trainers.YuccaLightningModule import YuccaLightningModule
from yucca.paths import yucca_results


class YuccaLightningManager:
    """
    The YuccaLightningWrapper is the one to rule them all.
    This will take ALL the arguments you need for training and apply them accordingly.

    First it automatically configure a host of parameters for you - such as batch size,
    patch size, modalities, classes, augmentations (and their hyperparameters) and formatting.
    It also instantiates the PyTorch Lightning trainer.

    Then, it will instantiate the YuccaLightningModule - containing the model, optimizers,
    loss functions and learning rates.

    Then, it will call the YuccaLightningDataModule. The DataModule creates the dataloaders that
    we use iterate over the chosen Task, which is wrapped in a YuccaDataset.

    YuccaLightningTrainer
    ├── model_params
    ├── data_params
    ├── aug_params
    ├── pl.Trainer
    |
    ├── YuccaLightningModule(model_params) -> model
    |   ├── network(model_params)
    |   ├── optim
    |   ├── loss_fn
    |   ├── scheduler
    |
    ├── YuccaLightningDataModule(data_params, aug_params) -> train_dataloader, val_dataloader
    |   ├── YuccaDataset(data_params, aug_params)
    |   ├── InfiniteRandomSampler
    |   ├── DataLoaders(YuccaDataset, InfiniteRandomSampler)
    |
    ├── pl.Trainer.fit(model, train_dataloader, val_dataloader)
    |
    """

    def __init__(
        self,
        ckpt_path: str = None,
        continue_training: bool = None,
        deep_supervision: bool = False,
        disable_logging: bool = False,
        folds: str = "0",
        max_epochs: int = 1000,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        num_workers: int = 8,
        planner: str = "YuccaPlanner",
        precision: str = "16-mixed",
        step_logging: bool = False,
        task: str = None,
        **kwargs,
    ):
        self.continue_training = continue_training
        self.ckpt_path = ckpt_path
        self.deep_supervision = deep_supervision
        self.disable_logging = disable_logging
        self.folds = folds
        self.max_epochs = max_epochs
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.name = self.__class__.__name__
        self.num_workers = num_workers
        self.planner = planner
        self.precision = precision
        self.step_logging = step_logging
        self.task = task
        self.kwargs = kwargs

        # default settings
        self.max_vram = 12
        self.is_initialized = False

        # Trainer settings
        self.train_batches_per_step = 250
        self.val_batches_per_step = 50
        self.trainer = L.Trainer

    def initialize(
        self,
        stage: Literal["fit", "test", "predict"],
        pred_data_dir: str = None,
        save_softmax: bool = False,
        segmentation_output_dir: str = "./",
    ):
        if stage == "fit":
            # do something training related
            # the stage param will disappear if nothing is found to be relevant here
            pass
        if stage == "test":
            raise NotImplementedError
        if stage == "predict":
            # do something inference related
            pass

        # Here we configure the outpath we will use to store model files and metadata
        # along with the path to plans file which will also be loaded.
        configurator = YuccaConfigurator(
            disable_logging=self.disable_logging,
            folds=self.folds,
            manager_name=self.name,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            planner=self.planner,
            segmentation_output_dir=segmentation_output_dir,
            save_softmax=save_softmax,
            tiny_patch=True if self.model_name == "TinyUNet" else False,
            task=self.task,
        )

        augmenter = YuccaAugmentationComposer(
            patch_size=configurator.patch_size,
            is_2D=True if self.model_dimensions == "2D" else False,
        )

        self.model_module = YuccaLightningModule(
            configurator=configurator,
            step_logging=self.step_logging,
            test_time_augmentation=bool(augmenter.mirror_p_per_sample),
        )

        self.data_module = YuccaDataModule(
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            configurator=configurator,
            num_workers=self.num_workers,
            pred_data_dir=pred_data_dir,
            pre_aug_patch_size=augmenter.pre_aug_patch_size,
        )

        self.trainer = L.Trainer(
            callbacks=configurator.callbacks,
            default_root_dir=configurator.outpath,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            logger=configurator.loggers,
            precision=self.precision,
            enable_progress_bar=not self.disable_logging,
            max_epochs=self.max_epochs,
            **self.kwargs,
        )

    def run_training(self):
        self.initialize(stage="fit")
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path="last",
        )

    def predict_folder(
        self,
        input_folder,
        output_folder: str = yucca_results,
        save_softmax=False,
    ):
        self.initialize(
            stage="predict",
            pred_data_dir=input_folder,
            segmentation_output_dir=output_folder,
            save_softmax=save_softmax,
        )
        self.trainer.predict(
            model=self.model_module,
            dataloaders=self.data_module,
            ckpt_path=self.ckpt_path,
        )


if __name__ == "__main__":
    # path = "/home/zcr545/YuccaData/yucca_models/Task001_OASIS/UNet__3D/YuccaPlanner/YuccaLightningManager/0/2023_11_08_15_19_14/checkpoints/test_ckpt.ckpt"
    path = None
    Manager = YuccaLightningManager(
        disable_logging=True,
        step_logging=True,
        ckpt_path=path,
        folds="0",
        model_name="TinyUNet",
        model_dimensions="2D",
        num_workers=0,
        task="Task001_OASIS",
    )

    Manager.run_training()
    # Manager.predict_folder(
    #    input_folder="/home/zcr545/YuccaData/yucca_raw_data/Task001_OASIS/imagesTs",
    #    output_folder="/home/zcr545/YuccaData/yucca_predictions",
    # )

# %%
