import lightning as L
from yucca.pipeline.configuration.configure_task import TaskConfig
from yucca.pipeline.configuration.configure_paths import get_path_config
from yucca.pipeline.configuration.configure_callbacks import get_callback_config
from yucca.pipeline.configuration.split_data import get_split_config
from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.modules.data.augmentation.augmentation_presets import generic
from repos.yucca.yucca.lightning_modules.BaseLightningModule import YuccaThinLightningModule
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.networks.networks import TinyUNet
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE


if __name__ == "__main__":
    config = {
        "batch_size": 2,
        "dims": "2D",
        "deep_supervision": False,
        "experiment": "default",
        "learning_rate": 1e-3,
        "loss_fn": DiceCE,
        "model": TinyUNet,
        "momentum": 0.99,
        "num_classes": 3,
        "num_modalities": 1,
        "patch_size": (32, 32),
        "plans_name": "demo",
        "plans": None,
        "split_idx": 0,
        "split_method": "kfold",
        "split_param": 5,
        "task": "Task001_OASIS",
        "task_type": "segmentation",
    }

    input_dims_config = InputDimensionsConfig(
        batch_size=config.get("batch_size"), patch_size=config.get("patch_size"), num_modalities=config.get("num_modalitites")
    )

    task_config = TaskConfig(
        task=config.get("task"),
        continue_from_most_recent=True,
        experiment=config.get("experiment"),
        manager_name="",
        model_dimensions=config.get("dims"),
        model_name=config.get("model_name"),
        patch_based_training=True,
        planner_name=config.get("plans_name"),
        split_idx=config.get("split_idx"),
        split_method=config.get("split_method"),
        split_param=config.get("split_param"),
    )

    path_config = get_path_config(task_config=task_config)

    split_config = get_split_config(method=task_config.split_method, param=task_config.split_param, path_config=path_config)

    callback_config = get_callback_config(
        save_dir=path_config.save_dir,
        version_dir=path_config.version_dir,
        experiment=task_config.experiment,
        version=path_config.version,
        enable_logging=False,
    )

    augmenter = YuccaAugmentationComposer(
        deep_supervision=config.get("deep_supervision"),
        patch_size=config.get("patch_size"),
        is_2D=True if config.get("dims") == "2D" else False,
        parameter_dict=generic,
        task_type_preset=config.get("task_type"),
    )

    model_module = YuccaThinLightningModule(
        model=config.get("model"),
        model_dimensions=config.get("model_dimensions"),
        num_classes=config.get("num_classes"),
        num_modalities=config.get("num_modalities"),
        patch_size=config.get("patch_size"),
        plans=config.get("plans"),
        deep_supervision=config.get("deep_supervision"),
        learning_rate=config.get("learning_rate"),
        loss_fn=config.get("loss_fn"),
        momentum=config.get("momentum"),
    )

    data_module = YuccaDataModule(
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        input_dims_config=input_dims_config,
        train_data_dir=path_config.train_data_dir,
        split_idx=task_config.split_idx,
        splits_config=split_config,
        task_type=config.get("task_type"),
    )

    trainer = L.Trainer(
        callbacks=callback_config.callbacks,
        default_root_dir=path_config.save_dir,
        limit_train_batches=2,
        limit_val_batches=2,
        log_every_n_steps=2,
        logger=callback_config.loggers,
        precision="32",
        profiler=callback_config.profiler,
        enable_progress_bar=True,
        max_epochs=2,
        accelerator="cpu",
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
        ckpt_path="last",
    )
