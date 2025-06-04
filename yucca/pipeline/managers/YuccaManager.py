import lightning as L
import torch
import wandb
import logging
import yucca
from batchgenerators.utilities.file_and_folder_operations import join
from typing import Literal, Union, Optional
from yucca.functional.utils.files_and_folders import recursive_find_python_class
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.pipeline.configuration.split_data import get_split_config, SplitConfig
from yucca.pipeline.configuration.configure_task import get_task_config
from yucca.pipeline.configuration.configure_callbacks import get_callback_config
from yucca.pipeline.configuration.configure_checkpoint import get_checkpoint_config
from yucca.pipeline.configuration.configure_seed import seed_everything_and_get_seed_config
from yucca.pipeline.configuration.configure_paths import get_path_config
from yucca.pipeline.configuration.configure_plans import get_plan_config
from yucca.pipeline.configuration.configure_input_dims import get_input_dims_config
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset, YuccaTestDataset, YuccaTestPreprocessedDataset
from yucca.modules.data.samplers import InfiniteRandomSampler
from yucca.modules.lightning_modules.YuccaLightningModule import YuccaLightningModule
from yucca.paths import get_results_path


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
        augmentation_params: dict = {},
        batch_size: Union[int, Literal["tiny"]] = None,
        ckpt_path: str = None,
        continue_from_most_recent: bool = True,
        deep_supervision: bool = False,
        enable_logging: bool = True,
        experiment: str = "default",
        learning_rate: float = 1e-3,
        loss: str = None,
        max_epochs: int = 1000,
        max_vram: int = 12,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        momentum: float = 0.9,
        num_workers: Optional[int] = None,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        optim_kwargs: dict = {},
        patch_based_training: bool = True,
        patch_size: Union[tuple, Literal["max", "min", "mean"]] = None,
        planner: str = "YuccaPlanner",
        precision: str = "bf16-mixed",
        profile: bool = False,
        p_oversample_foreground: Optional[float] = 0.33,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        split_idx: int = 0,
        split_data_method: str = "kfold",
        split_data_param: int = 5,
        step_logging: bool = False,
        task: str = None,
        train_batches_per_step: int = 250,
        use_label_regions: bool = False,
        val_batches_per_step: int = 50,
        wandb_log_model=False,
        **kwargs,
    ):
        self.ckpt_path = ckpt_path
        self.continue_from_most_recent = continue_from_most_recent
        self.deep_supervision = deep_supervision
        self.enable_logging = enable_logging
        self.experiment = experiment
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.max_epochs = max_epochs
        self.max_vram = max_vram
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.momentum = float(momentum)
        self.name = self.__class__.__name__
        self.num_workers = num_workers
        self.augmentation_params = augmentation_params
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optim_kwargs = optim_kwargs
        self.patch_based_training = patch_based_training
        self.patch_size = patch_size
        self.planner = planner
        self.precision = precision
        self.profile = profile
        self.p_oversample_foreground = p_oversample_foreground
        self.scheduler = scheduler
        self.split_idx = split_idx
        self.split_data_method = split_data_method
        self.split_data_param = split_data_param
        self.step_logging = step_logging
        self.task = task
        self.train_batches_per_step = train_batches_per_step
        self.use_label_regions = use_label_regions
        self.val_batches_per_step = val_batches_per_step
        self.wandb_log_model = wandb_log_model
        self.kwargs = kwargs

        # Automatically changes bfloat training if we're running on a GPU
        # that doesn't support it (otherwise it just crashes.)
        if "bf" in self.precision and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                self.precision = self.precision.replace("bf", "")

        # defaults
        self.data_module_class = YuccaDataModule
        self.lightning_module = YuccaLightningModule
        self.trainer = L.Trainer
        self.train_dataset_class = YuccaTrainDataset
        self.test_dataset_class = YuccaTestDataset
        self.accelerator = "cpu" if torch.backends.mps.is_available() and self.model_dimensions == "3D" else "auto"
        self.optim_kwargs.update({"lr": self.learning_rate, "momentum": self.momentum})

        if loss is None:
            self.loss = "SigmoidDiceBCE" if self.use_label_regions is True else "DiceCE"

        if self.kwargs.get("fast_dev_run"):
            self.setup_fast_dev_run()

    def initialize(
        self,
        stage: Literal["fit", "test", "predict"],
        disable_tta: bool = False,
        disable_inference_preprocessing: bool = False,
        pred_include_cases: list = None,
        overwrite_predictions: bool = False,
        pred_data_dir: str = None,
        save_softmax: bool = False,
        prediction_output_dir: str = "./",
    ):
        # Here we configure the outpath we will use to store model files and metadata
        # along with the path to plans file which will also be loaded.
        task_config = get_task_config(
            task=self.task,
            continue_from_most_recent=self.continue_from_most_recent,
            manager_name=self.name,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            patch_based_training=self.patch_based_training,
            planner_name=self.planner,
            experiment=self.experiment,
            split_idx=self.split_idx,
            split_data_method=self.split_data_method,
            split_data_param=self.split_data_param,
        )

        path_config = get_path_config(task_config=task_config, stage=stage)

        self.ckpt_config = get_checkpoint_config(
            ckpt_path=self.ckpt_path,
            continue_from_most_recent=task_config.continue_from_most_recent,
            current_experiment=task_config.experiment,
            path_config=path_config,
        )

        seed_config = seed_everything_and_get_seed_config(ckpt_seed=self.ckpt_config.ckpt_seed)

        self.plan_config = self.get_plan_config(
            ckpt_plans=self.ckpt_config.ckpt_plans,
            plans_path=path_config.plans_path,
            stage=stage,
            use_label_regions=self.use_label_regions,
        )

        callback_config = get_callback_config(
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
            save_multilabel_predictions=self.use_label_regions,
            wandb_log_model=self.wandb_log_model,
        )

        if stage == "fit":
            splits_config = get_split_config(task_config.split_method, task_config.split_param, path_config)
        else:
            splits_config = SplitConfig()

        input_dims_config = get_input_dims_config(
            plan=self.plan_config.plans,
            model_dimensions=task_config.model_dimensions,
            num_classes=self.plan_config.num_classes,
            ckpt_patch_size=self.ckpt_config.ckpt_patch_size,
            model_name=task_config.model_name,
            max_vram=self.max_vram,
            batch_size=self.batch_size,
            patch_based_training=task_config.patch_based_training,
            patch_size=self.patch_size,
        )

        augmenter = YuccaAugmentationComposer(
            deep_supervision=self.deep_supervision,
            patch_size=input_dims_config.patch_size,
            is_2D=True if self.model_dimensions == "2D" else False,
            parameter_dict=self.augmentation_params,
            task_type_preset=self.plan_config.task_type,
            labels=self.plan_config.labels,
            regions=self.plan_config.regions if self.plan_config.use_label_regions else None,
        )

        model, loss, preprocessor = self.find_classes_recursively(
            model=self.model_name, loss=self.loss, preprocessor=self.plan_config.plans["preprocessor"]
        )

        self.model_module = self.lightning_module(
            config=task_config.lm_hparams()
            | path_config.lm_hparams()
            | self.ckpt_config.lm_hparams()
            | seed_config.lm_hparams()
            | splits_config.lm_hparams()
            | self.plan_config.lm_hparams()
            | input_dims_config.lm_hparams()
            | callback_config.lm_hparams()
            | augmenter.lm_hparams(),
            model=model,
            deep_supervision=self.deep_supervision,
            disable_inference_preprocessing=disable_inference_preprocessing,
            loss_fn=loss,
            lr_scheduler=self.scheduler,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optim_kwargs,
            preprocessor=preprocessor,
            step_logging=self.step_logging,
            test_time_augmentation=not disable_tta if disable_tta is True else augmenter.mirror_p_per_sample > 0,
        )

        self.data_module = self.data_module_class(
            batch_size=input_dims_config.batch_size,
            patch_size=input_dims_config.patch_size,
            allow_missing_modalities=self.plan_config.allow_missing_modalities,
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            pred_include_cases=pred_include_cases,
            image_extension=self.plan_config.image_extension,
            overwrite_predictions=overwrite_predictions,
            num_workers=self.num_workers,
            pred_data_dir=pred_data_dir,
            pred_save_dir=prediction_output_dir,
            pre_aug_patch_size=augmenter.pre_aug_patch_size,
            p_oversample_foreground=self.p_oversample_foreground,
            splits_config=splits_config,
            split_idx=task_config.split_idx,
            task_type=self.plan_config.task_type,
            test_dataset_class=self.test_dataset_class,
            train_data_dir=path_config.train_data_dir,
            train_dataset_class=self.train_dataset_class,
        )

        self.verify_modules_are_valid()

        self.trainer = L.Trainer(
            accelerator=self.accelerator,
            callbacks=callback_config.callbacks,
            default_root_dir=path_config.save_dir,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            log_every_n_steps=min(self.train_batches_per_step, 50),
            logger=callback_config.loggers,
            precision=self.precision,
            profiler=callback_config.profiler,
            enable_progress_bar=not self.enable_logging,
            max_epochs=self.max_epochs,
            devices=1,
            **self.kwargs,
        )

    def run_training(self):
        self.initialize(stage="fit")
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path="last",
        )
        self.finish()

    def run_finetuning(self):
        self.initialize(stage="fit")
        self.model_module.load_state_dict(
            state_dict=torch.load(self.ckpt_config.ckpt_path, map_location=torch.device("cpu"), weights_only=False)[
                "state_dict"
            ],
            strict=False,
        )
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
        )
        self.finish()

    def predict_folder(
        self,
        input_folder,
        disable_tta: bool = False,
        disable_inference_preprocessing: bool = False,
        overwrite_predictions: bool = False,
        output_folder: str = get_results_path(),
        pred_include_cases: list = None,
        save_softmax=False,
    ):
        self.batch_size = 1
        self.initialize(
            stage="predict",
            disable_tta=disable_tta,
            disable_inference_preprocessing=disable_inference_preprocessing,
            pred_include_cases=pred_include_cases,
            overwrite_predictions=overwrite_predictions,
            pred_data_dir=input_folder,
            prediction_output_dir=output_folder,
            save_softmax=save_softmax,
        )

        self.trainer.predict(
            model=self.model_module,
            dataloaders=self.data_module,
            ckpt_path=self.ckpt_config.ckpt_path,
            return_predictions=False,
        )
        self.finish()

    def predict_preprocessed_folder(
        self,
        input_folder,
        disable_tta: bool = False,
        overwrite_predictions: bool = False,
        output_folder: str = get_results_path(),
        pred_include_cases: list = None,
        save_softmax=False,
    ):
        self.test_dataset_class = YuccaTestPreprocessedDataset
        self.predict_folder(
            input_folder=input_folder,
            disable_tta=disable_tta,
            disable_inference_preprocessing=True,
            overwrite_predictions=overwrite_predictions,
            output_folder=output_folder,
            pred_include_cases=pred_include_cases,
            save_softmax=save_softmax,
        )

    def find_classes_recursively(self, model=None, loss=None, preprocessor=None):
        if isinstance(model, str):
            model = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "modules", "networks")],
                class_name=model,
                current_module="yucca.modules.networks",
            )
        if isinstance(loss, str):
            loss = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "modules", "optimization", "loss_functions")],
                class_name=loss,
                current_module="yucca.modules.optimization.loss_functions",
            )
        if isinstance(preprocessor, str):
            preprocessor = recursive_find_python_class(
                folder=[join(yucca.__path__[0], "pipeline", "preprocessing")],
                class_name=preprocessor,
                current_module="yucca.pipeline.preprocessing",
            )
        return model, loss, preprocessor

    def finish(self):
        wandb.finish()

    def verify_modules_are_valid(self):
        # Method to expand for additional verifications
        self.verify_samplers_are_valid()

    def verify_samplers_are_valid(self):
        if (
            not issubclass(self.data_module.train_sampler, InfiniteRandomSampler) and self.train_batches_per_step is not None
        ) or (not issubclass(self.data_module.val_sampler, InfiniteRandomSampler) and self.val_batches_per_step is not None):
            logging.info(
                "Warning: you are limiting the amount of batches pr. step, but not sampling using InfiniteRandomSampler."
            )

    def setup_fast_dev_run(self):
        self.accelerator = "cpu"
        self.batch_size = 2
        self.patch_size = (32, 32)
        self.enable_logging = False
        self.model_dimensions = "2D"
        self.precision = 16
        self.train_batches_per_step = 10
        self.val_batches_per_step = 5

    @staticmethod
    def get_plan_config(ckpt_plans, plans_path, stage, use_label_regions):
        plan_config = get_plan_config(
            ckpt_plans=ckpt_plans,
            plans_path=plans_path,
            use_label_regions=use_label_regions,
            stage=stage,
        )
        return plan_config


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
