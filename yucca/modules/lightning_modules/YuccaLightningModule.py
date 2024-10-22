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
from yucca.functional.visualization import get_train_fig_with_inp_out_tar
from yucca.modules.lightning_modules.BaseLightningModule import BaseLightningModule
from yucca.functional.utils.torch_utils import measure_FLOPs
from fvcore.nn import flop_count_table
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE


class YuccaLightningModule(BaseLightningModule):
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
        self.task_type = config["task_type"]
        self.use_label_regions = "use_label_regions" in config.keys() and config["use_label_regions"]
        super().__init__(
            model=model,
            model_dimensions=config["model_dimensions"],
            num_classes=config["num_classes"],
            num_modalities=config["num_modalities"],
            patch_size=config["patch_size"],
            crop_to_nonzero=config["plans"]["crop_to_nonzero"],
            deep_supervision=deep_supervision,
            disable_inference_preprocessing=disable_inference_preprocessing,
            hparams_path=config["plans_path"],
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
            sliding_window_prediction=config["patch_based_training"],
            step_logging=step_logging,
            test_time_augmentation=test_time_augmentation,
            transpose_forward=list(map(int, config["plans"]["transpose_forward"])),
            transpose_backward=list(map(int, config["plans"]["transpose_backward"])),
        )
        self.config = config
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.save_hyperparameters(ignore=["model", "loss_fn", "lr_scheduler", "optimizer", "preprocessor"])

    def setup(self, stage):  # noqa: U100
        logging.info(f"Loading Model: {self.model_dimensions} {self.model.__name__}")
        if self.model_dimensions == "3D":
            conv_op = torch.nn.Conv3d
            norm_op = torch.nn.InstanceNorm3d
        else:
            conv_op = torch.nn.Conv2d
            norm_op = torch.nn.BatchNorm2d

        model_kwargs = {
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            "deep_supervision": self.deep_supervision,
            "norm_op": norm_op,
            # UNetR
            "patch_size": self.patch_size,
            # MedNeXt
            "checkpoint_style": "outside_block",
        }
        model_kwargs.update(self.model_kwargs)
        model_kwargs = filter_kwargs(self.model, model_kwargs)
        self.model = self.model(input_channels=self.num_modalities, num_classes=self.num_classes, **model_kwargs)
        self.visualize_model_with_FLOPs()

    def visualize_model_with_FLOPs(self):
        try:
            data = torch.randn((self.config["batch_size"], self.num_modalities, *self.patch_size))
            flops = measure_FLOPs(self.model, data)
            del data
            logging.info("\n" + flop_count_table(flops))
        except RuntimeError:
            logging.info("\n Model architecture could not be visualized.")

    def configure_metrics(self):
        if self.task_type == "classification":
            tmetrics_task = "multiclass" if self.num_classes > 2 else "binary"
            # can we get per-class?
            self.train_metrics = MetricCollection(
                {
                    "train/acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                    "train/roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "val/acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                    "val/roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
                }
            )

        if self.task_type == "segmentation":
            self.train_metrics = MetricCollection(
                {
                    "train/aggregated_dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        per_class=False,
                    ),
                    "train/mean_dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        average=True,
                    ),
                    "train/dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        per_class=True,
                    ),
                },
            )

            self.val_metrics = MetricCollection(
                {
                    "val/aggregated_dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        per_class=False,
                    ),
                    "val/mean_dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        average=True,
                    ),
                    "val/dice": GeneralizedDiceScore(
                        multilabel=self.use_label_regions,
                        num_classes=self.num_classes,
                        include_background=self.num_classes == 1 or self.use_label_regions,
                        weight_type="linear",
                        per_class=True,
                    ),
                },
            )

        if self.task_type == "self-supervised":
            self.train_metrics = MetricCollection({"train/MAE": MeanAbsoluteError()})
            self.val_metrics = MetricCollection({"train/MAE": MeanAbsoluteError()})

    def on_fit_start(self):
        if self.log_image_every_n_epochs is None:
            self.log_image_every_n_epochs = self.get_image_logging_epochs(self.trainer.max_epochs)

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
                task_type=self.task_type,
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
                task_type=self.task_type,
            )

    def configure_optimizers(self):
        # Configure the loss function
        loss_kwargs = filter_kwargs(self.loss_fn, self.loss_kwargs)
        self.loss_fn_train = self.loss_fn(**loss_kwargs)
        self.loss_fn_val = self.loss_fn(**loss_kwargs)
        if self.deep_supervision:
            self.loss_fn_train = DeepSupervisionLoss(self.loss_fn_train, weights=None)

        # Configure the optimizer
        optim_kwargs = {
            "momentum": 0.9,
            "eps": 1e-4,
            "weight_decay": 3e-5,
        }
        optim_kwargs.update(self.optim_kwargs)
        optim_kwargs = filter_kwargs(self.optim, optim_kwargs)
        self.optim = self.optim(self.model.parameters(), **optim_kwargs)

        # Configure the lr scheduler
        lr_scheduler_kwargs = {
            "T_max": self.trainer.max_epochs,
        }
        lr_scheduler_kwargs.update(self.lr_scheduler_kwargs)
        lr_scheduler_kwargs = filter_kwargs(self.lr_scheduler, lr_scheduler_kwargs)
        self.lr_scheduler = self.lr_scheduler(self.optim, **lr_scheduler_kwargs)

        # Finally return the optimizer and scheduler - the loss is not returned.
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def _log_dict_of_images_to_wandb(self, imagedict: {}, log_key: str, task_type: str = "segmentation"):
        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        case = os.path.splitext(os.path.split(imagedict["file_path"][batch_idx])[-1])[0]

        fig = get_train_fig_with_inp_out_tar(
            input=imagedict["input"][batch_idx],
            output=imagedict["output"][batch_idx],
            target=imagedict["target"][batch_idx],
            fig_title=case,
            task_type=task_type,
        )
        wandb.log({log_key: wandb.Image(fig)}, commit=False)
        plt.close(fig)

    @property
    def log_image_this_epoch(self):
        if isinstance(self.log_image_every_n_epochs, int):
            return self.current_epoch % self.log_image_every_n_epochs == 0
        if isinstance(self.log_image_every_n_epochs, list):
            return self.current_epoch in self.log_image_every_n_epochs

    @staticmethod
    def get_image_logging_epochs(final_epoch: int = 1000):
        first_half = np.logspace(0, 5, 10, base=4, endpoint=False)
        second_half = final_epoch - np.logspace(0, 5, 10, base=4, endpoint=False)[::-1]
        indices = sorted(np.concatenate((first_half, second_half)).astype(int))
        return indices


if __name__ == "__main__":
    f = YuccaLightningModule(
        config={
            "model_name": "TinyUNet",
            "plans": {"preprocessor": "YuccaPreprocessor"},
            "model_dimensions": "3D",
            "num_classes": 2,
            "num_modalities": 1,
            "patch_size": (32, 32, 32),
            "plans_path": "",
            "patch_based_training": True,
            "task_type": "segmentation",
        },
    )
    data = torch.randn((2, 1, *(32, 32, 32)))
    f.setup(stage="test")
    f.forward(data)
