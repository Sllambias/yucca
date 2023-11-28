import lightning as L
import torch
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
    The YuccaLightningManager class provides a convenient way to manage the training and inference processes in the Yucca project.
    It encapsulates the configuration, setup, and execution steps, making it easier to conduct experiments and predictions with consistent settings.

    The initialize method of the YuccaLightningManager class is responsible for setting up the necessary components for either training or inference.
    This method performs the configuration of paths, creates data augmentation objects, and sets up the PyTorch Lightning modules (model and data module)
    based on the specified stage in the pipeline. The stages can be "fit" (training), "test" (testing), or "predict" (inference).

    It performs the following steps:
        (1) Configure Paths: The YuccaConfigurator class is used to set up paths for training,
        including the training data directory, output path, and plans file path.

        (2) Create Augmentation Objects: The YuccaAugmentationComposer class is used to create data augmentation objects
        (augmenter.train_transforms and augmenter.val_transforms) based on the specified patch size and model dimensionality.

        (3) Set Up PyTorch Lightning Modules: The YuccaLightningModule class is initialized with the appropriate configuration
        using the YuccaConfigurator object. This module is responsible for defining the neural network architecture, loss functions,
        and optimization strategies.

        (4) Set Up PyTorch Lightning Data Module: The YuccaDataModule class is initialized with the composed training and validation transforms,
        along with the YuccaConfigurator object. This module handles the loading and preprocessing of the training, validation, and prediction datasets.

        (5) Initialize PyTorch Lightning Trainer: The PyTorch Lightning Trainer is configured with the necessary settings, including callbacks,
        output directory, precision, and other specified parameters.
    """

    def __init__(
        self,
        ckpt_path: str = None,
        continue_from_most_recent: bool = True,
        deep_supervision: bool = False,
        disable_logging: bool = False,
        folds: str = "0",
        max_epochs: int = 1000,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        num_workers: int = 8,
        planner: str = "YuccaPlanner",
        precision: str = "16-mixed",
        profile: bool = False,
        step_logging: bool = False,
        task: str = None,
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.continue_from_most_recent = continue_from_most_recent
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
        self.profile = profile
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
        disable_tta: bool = False,
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
            continue_from_most_recent=self.continue_from_most_recent,
            disable_logging=self.disable_logging,
            folds=self.folds,
            manager_name=self.name,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            planner=self.planner,
            profile=self.profile,
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
            test_time_augmentation=not disable_tta if disable_tta is True else bool(augmenter.mirror_p_per_sample),
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
            profiler=configurator.profiler,
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
        disable_tta: bool = False,
        output_folder: str = yucca_results,
        save_softmax=False,
    ):
        self.initialize(
            stage="predict",
            disable_tta=disable_tta,
            pred_data_dir=input_folder,
            segmentation_output_dir=output_folder,
            save_softmax=save_softmax,
        )
        with torch.inference_mode():
            self.trainer.predict(
                model=self.model_module,
                dataloaders=self.data_module,
                ckpt_path=self.ckpt_path,
            )


if __name__ == "__main__":
    # path = "/home/zcr545/YuccaData/yucca_models/Task001_OASIS/UNet__3D/YuccaPlanner/YuccaLightningManager/0/2023_11_08_15_19_14/checkpoints/test_ckpt.ckpt"
    path = None
    Manager = YuccaLightningManager(
        disable_logging=False,
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
