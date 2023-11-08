# %%
import lightning as L
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
        continue_training: bool = None,
        deep_supervision: bool = False,
        folds: int = 0,
        model_dimensions: str = "3D",
        model_name: str = "TinyUNet",
        planner: str = "YuccaPlanner",
        precision: str = "16-mixed",
        task: str = None,
        ckpt_path: str = None,
        **kwargs,
    ):
        self.continue_training = continue_training
        self.ckpt_path = ckpt_path
        self.deep_supervision = deep_supervision
        self.folds = folds
        self.model_dimensions = model_dimensions
        self.model_name = model_name
        self.name = self.__class__.__name__
        self.planner = planner
        self.precision = precision
        self.task = task
        self.kwargs = kwargs

        # default settings
        self.max_vram = 2
        self.is_initialized = False

        # Trainer settings
        self.max_epochs = 5
        self.train_batches_per_step = 250
        self.val_batches_per_step = 50
        self.trainer = L.Trainer

    def initialize(
        self,
        train=False,
        pred_data_dir: str = None,
        segmentation_output_dir: str = "./",
    ):
        # Here we configure the outpath we will use to store model files and metadata
        # along with the path to plans file which will also be loaded.
        print("initializing, train is:", train)
        configurator = YuccaConfigurator(
            folds=self.folds,
            model_dimensions=self.model_dimensions,
            model_name=self.model_name,
            segmentation_output_dir=segmentation_output_dir,
            planner=self.planner,
            task=self.task,
        )

        augmenter = YuccaAugmentationComposer(
            patch_size=configurator.patch_size,
            is_2D=True if self.model_dimensions == "2D" else False,
        )

        self.model_module = YuccaLightningModule(
            num_classes=configurator.num_classes,
            num_modalities=configurator.num_modalities,
            patch_size=configurator.patch_size,
            plans_path=configurator.plans_path,
            test_time_augmentation=bool(augmenter.mirror_p_per_sample),
        )

        self.data_module = YuccaDataModule(
            configurator=configurator,
            pred_data_dir=pred_data_dir,
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
        )

        self.trainer = L.Trainer(
            callbacks=configurator.callbacks,
            default_root_dir=configurator.outpath,
            limit_train_batches=self.train_batches_per_step,
            limit_val_batches=self.val_batches_per_step,
            logger=configurator.loggers,
            precision=self.precision,
            enable_progress_bar=False,
            max_epochs=self.max_epochs,
            **self.kwargs,
        )

    def run_training(self):
        self.initialize(train=True)
        self.trainer.fit(
            model=self.model_module,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def predict_folder(
        self,
        input_folder,
        output_folder: str = yucca_results,
        not_strict=True,
        save_softmax=False,
        overwrite=False,
        do_tta=False,
    ):
        self.initialize(
            train=False,
            pred_data_dir=input_folder,
            segmentation_output_dir=output_folder,
        )
        self.trainer.predict(
            model=self.model_module,
            dataloaders=self.data_module,
            ckpt_path=self.ckpt_path,
        )


if __name__ == "__main__":
    import warnings
    import logging

    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    path = None
    Manager = YuccaLightningManager(
        task="Task001_OASIS",
        ckpt_path=path,
    )
    # Manager.initialize()
    Manager.run_training()


# %%
