import lightning as L
import torch
from typing import Literal, Union, Optional
from yucca.training.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.training.configuration.split_data import get_split_config
from yucca.training.configuration.configure_task import get_task_config
from yucca.training.configuration.configure_callbacks import get_callback_config
from yucca.training.configuration.configure_paths_and_version import get_path_and_version_config
from yucca.training.configuration.configure_plans import get_plan_config
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.lightning_modules.YuccaLightningModule import YuccaLightningModule
from yucca.training.configuration.input_dimensions import get_input_dims
from yucca.paths import yucca_results


class YuccaManager:
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
        loss: str = None,
        max_epochs: int = 1000,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        num_workers: int = 8,
        patch_size: Union[tuple, Literal["max", "min", "mean"]] = None,
        planner: str = "YuccaPlanner",
        precision: str = "16-mixed",
        profile: bool = False,
        split_idx: int = 0,
        step_logging: bool = False,
        task: str = None,
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.continue_from_most_recent = continue_from_most_recent
        self.deep_supervision = deep_supervision
        self.disable_logging = disable_logging
        self.loss = loss
        self.max_epochs = max_epochs
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.name = self.__class__.__name__
        self.num_workers = num_workers
        self.planner = planner
        self.precision = precision
        self.profile = profile
        self.split_idx = split_idx
        self.step_logging = step_logging
        self.task = task
        self.kwargs = kwargs

        if patch_size is None:
            self.patch_size = "tiny" if self.model_name == "TinyUNet" else None
        else:
            self.patch_size = patch_size

        # default settings
        self.max_vram = 12

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
        prediction_output_dir: str = "./",
    ):
        # Here we configure the outpath we will use to store model files and metadata
        # along with the path to plans file which will also be loaded.
        task_config = get_task_config(
            continue_from_most_recent=self.continue_from_most_recent,
            manager_name=self.name,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            planner_name=self.planner,
            split_idx=self.split_idx,
            task=self.task,
        )

        self.path_config = get_path_and_version_config(
            continue_from_most_recent=task_config.continue_from_most_recent,
            manager_name=task_config.manager_name,
            model_dimensions=task_config.model_dimensions,
            model_name=task_config.model_name,
            planner_name=task_config.planner_name,
            split_idx=task_config.split_idx,
            task=task_config.task,
        )

        plan_config = get_plan_config(
            ckpt_path=self.ckpt_path,
            continue_from_most_recent=task_config.continue_from_most_recent,
            plans_path=self.path_config.plans_path,
            version=self.path_config.version,
            version_dir=self.path_config.version_dir,
        )

        splits = get_split_config(train_data_dir=self.path_config.train_data_dir, task=task_config.task)
        input_dims = get_input_dims(
            plan=plan_config.plans,
            model_dimensions=task_config.model_dimensions,
            num_classes=plan_config.num_classes,
            model_name=task_config.model_name,
            max_vram=self.max_vram,
            patch_size=self.patch_size,
        )

        augmenter = YuccaAugmentationComposer(
            patch_size=input_dims.patch_size,
            is_2D=True if self.model_dimensions == "2D" else False,
            use_preset_for_task_type=plan_config.task_type,
        )

        callback_config = get_callback_config(
            task=task_config.task,
            save_dir=self.path_config.save_dir,
            version_dir=self.path_config.version_dir,
            version=self.path_config.version,
            disable_logging=self.disable_logging,
            prediction_output_dir=prediction_output_dir,
            profile=self.profile,
            save_softmax=save_softmax,
        )
        self.model_module = YuccaLightningModule(
            config=task_config.lm_hparams()
            | self.path_config.lm_hparams()
            | plan_config.lm_hparams()
            | splits.lm_hparams()
            | input_dims.lm_hparams(),
            loss_fn=self.loss,
            stage=stage,
            step_logging=self.step_logging,
            test_time_augmentation=not disable_tta if disable_tta is True else bool(augmenter.mirror_p_per_sample),
        )

        self.data_module = YuccaDataModule(
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            input_dims=input_dims,
            num_workers=self.num_workers,
            plan_config=plan_config,
            pred_data_dir=pred_data_dir,
            pre_aug_patch_size=augmenter.pre_aug_patch_size,
            splits=splits,
            split_idx=task_config.split_idx,
            train_data_dir=self.path_config.train_data_dir,
        )

        self.trainer = L.Trainer(
            callbacks=callback_config.callbacks,
            default_root_dir=self.path_config.save_dir,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            logger=callback_config.loggers,
            precision=self.precision,
            profiler=callback_config.profiler,
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

    def run_finetuning(self):
        self.initialize(stage="fit")
        self.model_module.load_state_dict(
            torch.load(self.path_config.ckpt_path, map_location=torch.device("cpu"))["state_dict"], strict=False
        )
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
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
            prediction_output_dir=output_folder,
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
    Manager = YuccaManager(
        disable_logging=False,
        ckpt_path=path,
        model_name="TinyUNet",
        model_dimensions="2D",
        num_workers=0,
        split_idx=0,
        step_logging=True,
        task="Task001_OASIS",
    )
    Manager.initialize("fit")
    # Manager.run_training()
    # Manager.predict_folder(
    #    input_folder="/home/zcr545/YuccaData/yucca_raw_data/Task001_OASIS/imagesTs",
    #    output_folder="/home/zcr545/YuccaData/yucca_predictions",
    # )