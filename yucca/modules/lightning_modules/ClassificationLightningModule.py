import torch
import wandb
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError
from yucca.modules.optimization.loss_functions.deep_supervision import DeepSupervisionLoss
from yucca.functional.utils.kwargs import filter_kwargs
from yucca.modules.metrics.training_metrics import Accuracy, AUROC, GeneralizedDiceScore
from yucca.modules.lightning_modules.YuccaLightningModule import YuccaLightningModule
from yucca.functional.utils.torch_utils import measure_FLOPs
from fvcore.nn import flop_count_table
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE

from yucca.functional.visualization import get_cls_train_fig_with_inp_out_tar


class ClassificationLightningModule(YuccaLightningModule):
    """
    The YuccaLightningModule class is an implementation of the PyTorch Lightning module designed for the Yucca project.
    It extends the LightningModule class and encapsulates the neural network model, loss functions, and optimization logic.
    This class is responsible for handling training, validation, and inference steps within the Yucca machine learning pipeline.
    """

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        deep_supervision: bool = False,
        disable_inference_preprocessing: bool = False,
        loss_fn: torch.nn.Module = DiceCE,
        loss_kwargs: dict = {
            "soft_dice_kwargs": {"apply_softmax": True},
        },
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs: dict = {
            "eta_min": 1e-9,
        },
        model_kwargs: dict = {},
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        optimizer_kwargs={
            "lr": 1e-3,
        },
        sliding_window_overlap: float = 0.5,
        step_logging: bool = False,
        test_time_augmentation: bool = False,
        preprocessor=None,
        progress_bar: bool = False,
        log_image_every_n_epochs: int = None,
    ):
        super().__init__(
            config=config,
            model=model,
            deep_supervision=deep_supervision,
            disable_inference_preprocessing=disable_inference_preprocessing,
            loss_fn=loss_fn,
            loss_kwargs=loss_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            model_kwargs=model_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            preprocessor=preprocessor,
            progress_bar=progress_bar,
            sliding_window_overlap=sliding_window_overlap,
            step_logging=step_logging,
            test_time_augmentation=test_time_augmentation,
        )

        self.config = config
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.get_train_fig_fn = get_cls_train_fig_with_inp_out_tar
        self.save_hyperparameters(ignore=["model", "loss_fn", "lr_scheduler", "optimizer", "preprocessor"])

    def setup(self, stage):  # noqa: U100
        logging.info(f"Loading Model: {self.model_dimensions} {self.model.__name__}")
        if self.model_dimensions == "3D":
            conv_op = torch.nn.Conv3d
            norm_op = torch.nn.InstanceNorm3d
        else:
            conv_op = torch.nn.Conv2d
            norm_op = torch.nn.BatchNorm2d

        model_kwargs = {}
        model_kwargs.update(self.model_kwargs)
        model_kwargs = filter_kwargs(self.model, model_kwargs)
        self.model = self.model(input_channels=self.num_modalities, num_classes=self.num_classes, **model_kwargs)
        self.visualize_model_with_FLOPs()

    def configure_metrics(self):
        tmetrics_task = "multiclass"  # if self.num_classes > 2 else "binary"
        self.train_metrics = MetricCollection(
            {
                "train/acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                # "train/roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val/acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                # "val/roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
            }
        )

    def training_step(self, batch, batch_idx):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        metrics = self.compute_metrics(self.train_metrics, output, target, ignore_index=None)
        self.log_dict(
            {"train/loss": loss} | metrics,
            on_step=self.step_logging,
            on_epoch=self.epoch_logging,
            prog_bar=self.progress_bar,
            logger=True,
        )

        if batch_idx == 0 and wandb.run is not None and self.log_image_this_epoch is True:
            self._log_dict_of_images_to_wandb(
                {
                    "input": inputs.detach().cpu().to(torch.float32).numpy(),
                    "target": target.detach().cpu().to(torch.float32).numpy(),
                    "output": output.detach().cpu().to(torch.float32).numpy(),
                    "file_path": file_path,
                },
                log_key="train",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        output = self(inputs)

        loss = self.loss_fn_val(output, target)

        metrics = self.compute_metrics(self.val_metrics, output, target, ignore_index=None)
        self.log_dict(
            {"val/loss": loss} | metrics,
            on_step=self.step_logging,
            on_epoch=self.epoch_logging,
            prog_bar=self.progress_bar,
            logger=True,
        )

        if batch_idx == 0 and wandb.run is not None and self.log_image_this_epoch is True:
            self._log_dict_of_images_to_wandb(
                {
                    "input": inputs.detach().cpu().to(torch.float32).numpy(),
                    "target": target.detach().cpu().to(torch.float32).numpy(),
                    "output": output.detach().cpu().to(torch.float32).numpy(),
                    "file_path": file_path,
                },
                log_key="val",
            )
