import lightning as L
import torch
from typing import Literal, Union, Optional
from yucca.training.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.training.configuration.split_data import get_split_config
from yucca.training.configuration.configure_task import get_task_config
from yucca.training.configuration.configure_callbacks import get_callback_config
from yucca.training.configuration.configure_checkpoint import get_checkpoint_config
from yucca.training.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.training.configuration.configure_paths import get_path_config
from yucca.training.configuration.configure_plans import get_plan_config
from yucca.training.configuration.configure_input_dims import get_input_dims_config
from yucca.training.data_loading.YuccaDataModule import YuccaDataModule
from yucca.training.lightning_modules.YuccaLightningModule import YuccaLightningModule
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
        enable_logging: bool = True,
        learning_rate: float = 1e-3,
        loss: str = None,
        max_epochs: int = 1000,
        max_vram: int = 12,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        num_workers: int = 8,
        patch_based_training: bool = True,
        patch_size: Union[tuple, Literal["max", "min", "mean"]] = None,
        planner: str = "YuccaPlanner",
        precision: str = "bf16-mixed",
        profile: bool = False,
        split_idx: int = 0,
        step_logging: bool = False,
        task: str = None,
        experiment: str = "default",
        train_batches_per_step: int = 250,
        val_batches_per_step: int = 50,
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.continue_from_most_recent = continue_from_most_recent
        self.deep_supervision = deep_supervision
        self.enable_logging = enable_logging
        self.experiment = experiment
        self.loss = loss
        self.max_epochs = max_epochs
        self.max_vram = max_vram
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.name = self.__class__.__name__
        self.num_workers = num_workers
        self.patch_based_training = patch_based_training
        self.patch_size = patch_size
        self.planner = planner
        self.precision = precision
        self.profile = profile
        self.split_idx = split_idx
        self.step_logging = step_logging
        self.task = task
        self.train_batches_per_step = train_batches_per_step
        self.val_batches_per_step = val_batches_per_step
        self.kwargs = kwargs

        # Configure basic parameters
        if self.patch_size is None and self.model_name == "TinyUNet":
            self.patch_size = "tiny"

        # Automatically changes bfloat training if we're running on a GPU
        # that doesn't support it (otherwise it just crashes.)
        if "bf" in self.precision and not torch.cuda.is_bf16_supported():
            self.precision = self.precision.replace("bf", "")

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
            patch_based_training=self.patch_based_training,
            planner_name=self.planner,
            split_idx=self.split_idx,
            task=self.task,
            experiment=self.experiment,
        )

        path_config = get_path_config(task_config=task_config)

        self.ckpt_config = get_checkpoint_config(
            ckpt_path=self.ckpt_path,
            continue_from_most_recent=task_config.continue_from_most_recent,
            current_experiment=task_config.experiment,
            path_config=path_config,
        )

        seed_config = seed_everything_and_get_seed_config(ckpt_seed=self.ckpt_config.ckpt_seed)

        plan_config = get_plan_config(
            ckpt_plans=self.ckpt_config.ckpt_plans,
            continue_from_most_recent=task_config.continue_from_most_recent,
            plans_path=path_config.plans_path,
            stage=stage,
        )

        splits_config = get_split_config(train_data_dir=path_config.train_data_dir, task=task_config.task)

        input_dims_config = get_input_dims_config(
            plan=plan_config.plans,
            model_dimensions=task_config.model_dimensions,
            num_classes=plan_config.num_classes,
            model_name=task_config.model_name,
            max_vram=self.max_vram,
            patch_based_training=task_config.patch_based_training,
            patch_size=self.patch_size,
        )

        augmenter = YuccaAugmentationComposer(
            deep_supervision=self.deep_supervision,
            patch_size=input_dims_config.patch_size,
            is_2D=True if self.model_dimensions == "2D" else False,
            task_type_preset=plan_config.task_type,
        )

        callback_config = get_callback_config(
            task=task_config.task,
            model_name=task_config.model_name,
            save_dir=path_config.save_dir,
            version_dir=path_config.version_dir,
            ckpt_version_dir=self.ckpt_config.ckpt_version_dir,
            ckpt_wandb_id=self.ckpt_config.ckpt_wandb_id,
            experiment=task_config.experiment,
            version=path_config.version,
            enable_logging=self.enable_logging,
            log_lr=True,
            prediction_output_dir=prediction_output_dir,
            profile=self.profile,
            save_softmax=save_softmax,
        )

        self.model_module = YuccaLightningModule(
            config=task_config.lm_hparams()
            | path_config.lm_hparams()
            | self.ckpt_config.lm_hparams()
            | seed_config.lm_hparams()
            | splits_config.lm_hparams()
            | plan_config.lm_hparams()
            | input_dims_config.lm_hparams()
            | callback_config.lm_hparams(),
            deep_supervision=self.deep_supervision,
            loss_fn=self.loss,
            stage=stage,
            step_logging=self.step_logging,
            test_time_augmentation=not disable_tta if disable_tta is True else bool(augmenter.mirror_p_per_sample),
        )

        self.data_module = YuccaDataModule(
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            input_dims_config=input_dims_config,
            num_workers=self.num_workers,
            plan_config=plan_config,
            pred_data_dir=pred_data_dir,
            pre_aug_patch_size=augmenter.pre_aug_patch_size,
            splits_config=splits_config,
            split_idx=task_config.split_idx,
            train_data_dir=path_config.train_data_dir,
        )

        self.trainer = L.Trainer(
            callbacks=callback_config.callbacks,
            default_root_dir=path_config.save_dir,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            logger=callback_config.loggers,
            precision=self.precision,
            profiler=callback_config.profiler,
            enable_progress_bar=not self.enable_logging,
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
            state_dict=torch.load(self.ckpt_config.ckpt_path, map_location=torch.device("cpu"))["state_dict"], strict=False
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
                ckpt_path=self.ckpt_config.ckpt_path,
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
