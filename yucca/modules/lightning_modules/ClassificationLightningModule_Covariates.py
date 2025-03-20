from yucca.modules.lightning_modules.ClassificationLightningModule import ClassificationLightningModule
from yucca.functional.preprocessing import reverse_preprocessing
import wandb
import torch
import logging
from yucca.functional.utils.torch_utils import measure_FLOPs
from fvcore.nn import flop_count_table


class ClassificationLightningModule_Covariates(ClassificationLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, cov):
        return self.model(inputs, cov)

    def training_step(self, batch, batch_idx):
        inputs, cov, target, file_path = batch["image"], batch["covariates"], batch["label"], batch["file_path"]
        output = self(inputs, cov)
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
        inputs, cov, target, file_path = batch["image"], batch["covariates"], batch["label"], batch["file_path"]
        output = self(inputs, cov)

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # noqa: U100
        logits = self.model.predict(data=batch["data"], cov=batch["covariates"])
        if self.disable_inference_preprocessing:
            logits, data_properties = reverse_preprocessing(
                crop_to_nonzero=self.crop_to_nonzero,
                images=logits,
                image_properties=batch["data_properties"],
                n_classes=self.num_classes,
                transpose_forward=self.transpose_forward,
                transpose_backward=self.transpose_backward,
            )
        else:
            logits, data_properties = self.preprocessor.reverse_preprocessing(
                logits, batch["data_properties"], num_classes=self.num_classes
            )
        return {"logits": logits, "properties": data_properties, "case_id": batch["case_id"]}

    def visualize_model_with_FLOPs(self):
        try:
            data = torch.randn((self.config["batch_size"], self.num_modalities, *self.patch_size))
            cov = torch.randn((2))
            flops = measure_FLOPs(self.model, (data, cov))
            del data
            logging.info("\n" + flop_count_table(flops))
        except RuntimeError:
            logging.info("\n Model architecture could not be visualized.")
