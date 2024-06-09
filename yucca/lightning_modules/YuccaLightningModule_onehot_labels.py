import wandb
import torch
from torchmetrics import MetricCollection
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.classification import MulticlassF1Score
from yucca.lightning_modules.YuccaLightningModule import YuccaLightningModule


class YuccaLightningModule_onehot_labels(YuccaLightningModule):
    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ):
        super().__init__(
            config,
            *args,
            **kwargs,
        )
        # self.regions_labeled = config["regions_labeled"] currently not used, but can be used during inference to go from regions -> labels

        self.train_metrics = MetricCollection(
            {
                "train/dice": GeneralizedDiceScore(num_classes=self.num_classes),
                "train/F1": MulticlassF1Score(num_classes=self.num_classes),
            },
        )

        self.val_metrics = MetricCollection(
            {
                "val/dice": GeneralizedDiceScore(num_classes=self.num_classes),
                "val/F1": MulticlassF1Score(num_classes=self.num_classes),
            },
        )

        self.loss_fn = "SigmoidDiceBCE"

    def training_step(self, batch, batch_idx):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        output = (torch.sigmoid(output) > 0.5).long()

        metrics = self.compute_metrics(self.train_metrics, output, target)
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

        output = (torch.sigmoid(output) > 0.5).long()

        metrics = self.compute_metrics(self.val_metrics, output, target)
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
