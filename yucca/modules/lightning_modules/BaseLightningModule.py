import lightning as L
import torch
import logging
import copy
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice
from yucca.pipeline.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.modules.optimization.loss_functions.deep_supervision import DeepSupervisionLoss
from yucca.modules.metrics.training_metrics import F1
from yucca.modules.optimization.loss_functions.nnUNet_losses import DiceCE
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from yucca.functional.preprocessing import reverse_preprocessing


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        model_dimensions: str,
        num_classes: int,
        num_modalities: int,
        patch_size: tuple,
        plans: dict,
        deep_supervision: bool = False,
        disable_inference_preprocessing: bool = False,
        hparams_path: str = None,
        loss_fn: torch.nn.Module = DiceCE,
        loss_kwargs: dict = {
            "soft_dice_kwargs": {"apply_softmax": True},
        },
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_kwargs: dict = {
            "T_max": 1000,
            "eta_min": 1e-9,
        },
        model_kwargs: dict = {},
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        optimizer_kwargs: dict = {
            "lr": 1e-3,
            "momentum": 0.9,
            "eps": 1e-4,
            "weight_decay": 3e-5,
        },
        preprocessor: YuccaPreprocessor = None,
        progress_bar: bool = False,
        sliding_window_overlap: float = 0.5,
        sliding_window_prediction: bool = True,
        step_logging: bool = False,
        test_time_augmentation: bool = False,
    ):
        super().__init__()
        # Extract parameters from the configurator
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.hparams_path = hparams_path
        self.plans = plans
        self.model = model
        self.model_dimensions = model_dimensions
        self.model_kwargs = model_kwargs
        self.patch_size = patch_size
        self.preprocessor = preprocessor
        self.sliding_window_prediction = sliding_window_prediction

        # Loss, optimizer and scheduler parameters
        self.deep_supervision = deep_supervision
        self.disable_inference_preprocessing = disable_inference_preprocessing
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs
        self.optim = optimizer
        self.optim_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # Inference
        self.sliding_window_overlap = sliding_window_overlap
        self.test_time_augmentation = test_time_augmentation

        # Evaluation and logging
        if step_logging is True:  # Blame PyTorch lightning for this war crime
            self.step_logging = None
            self.epoch_logging = None
        else:
            self.step_logging = False
            self.epoch_logging = True

        self.progress_bar = progress_bar

        logging.info(f"{self.__class__.__name__} initialized")
        logging.info(f"Deep Supervision Enabled: {self.deep_supervision}")

        # If we are training we save params and then start training
        # Do not overwrite parameters during inference.
        self.save_hyperparameters(ignore=["model", "loss_fn", "lr_scheduler", "optimizer", "preprocessor"])

    def setup(self, stage):  # noqa: U100
        logging.info(f"Loading Model: {self.model_dimensions} {self.model.__class__.__name__}")
        self.model = self.model(input_channels=self.num_modalities, num_classes=self.num_classes, **self.model_kwargs)

    def compute_metrics(self, metrics, output, target, ignore_index: int = 0):
        metrics = metrics(output, target)
        tmp = {}
        to_drop = []
        for key in metrics.keys():
            if metrics[key].numel() > 1:
                to_drop.append(key)
                for i, val in enumerate(metrics[key]):
                    if not i == ignore_index:
                        tmp[key + "_" + str(i)] = val
        for k in to_drop:
            metrics.pop(k)
        metrics.update(tmp)
        return metrics

    def configure_optimizers(self):
        self.loss_fn_train = self.loss_fn(**self.loss_kwargs)
        self.loss_fn_val = self.loss_fn(**self.loss_kwargs)

        # If deep_supervision is enabled we wrap our training loss (and potentially specify weights)
        # We leave the validation loss as is, as deep_supervision is not used for validation.
        if self.deep_supervision:
            self.loss_fn_train = DeepSupervisionLoss(self.loss_fn_train, weights=None)

        self.optim = self.optim(self.model.parameters(), **self.optim_kwargs)
        self.lr_scheduler = self.lr_scheduler(self.optim, **self.lr_scheduler_kwargs)
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def forward(self, inputs):
        return self.model(inputs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # First we filter out layers that have changed in size
        # This is often the case in the output layer.
        # If we are finetuning on a task with a different number of classes
        # than the pretraining task, the # output channels will have changed.
        old_params = copy.deepcopy(self.state_dict())
        state_dict = {
            k: v for k, v in state_dict.items() if (k in old_params) and (old_params[k].shape == state_dict[k].shape)
        }
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        rejected_keys_shape = [k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape]
        rejected_keys_data = []

        # Here there's also potential to implement custom loading functions.
        # E.g. to load 2D pretrained models into 3D by repeating or something like that.

        # Now keep track of the # of layers with succesful weight transfers
        successful = 0
        unsuccessful = 0
        super().load_state_dict(state_dict, *args, **kwargs)
        new_params = self.state_dict()
        for param_name, p1, p2 in zip(old_params.keys(), old_params.values(), new_params.values()):
            # If more than one param in layer is NE (not equal) to the original weights we've successfully loaded new weights.
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if param_name not in rejected_keys_new and param_name not in rejected_keys_shape:
                    rejected_keys_data.append(param_name)

        logging.warn(f"Succesfully transferred weights for {successful}/{successful+unsuccessful} layers")
        logging.warn(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )

        return successful

    def on_fit_start(self):
        self.train_metrics = MetricCollection(
            {
                "train/dice": Dice(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None),
                "train/F1": F1(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None, average=None),
            },
        )
        self.val_metrics = MetricCollection(
            {
                "val/dice": Dice(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None),
                "val/F1": F1(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None, average=None),
            },
        )

    def on_predict_start(self):
        if self.disable_inference_preprocessing:
            self.predict = self.predict_without_preprocessing
        else:
            self.preprocessor = self.preprocessor(self.hparams_path)
            self.predict = self.predict_with_preprocessing

    def predict_step(self, batch, batch_idx, dataloader_idx=0):  # noqa: U100
        case, case_id = batch
        logits, case_properties = self.predict(case)
        return {"logits": logits, "properties": case_properties, "case_id": case_id[0]}

    def predict_with_preprocessing(self, case):
        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(case, self.patch_size, self.sliding_window_prediction)
        logits = self._predict(case_preprocessed)

        logits, case_properties = self.preprocessor.reverse_preprocessing(logits, case_properties)
        return logits, case_properties

    def predict_without_preprocessing(self, case):
        # First split the list into case and case_properties
        case, case_properties = case
        # Load and unpack from tuple if wrapped (as will be the case by standard DataLoaders)
        case = torch.load(case[0]) if isinstance(case, tuple) else torch.load(case, "r")
        case_properties = (
            load_pickle(case_properties[0]) if isinstance(case_properties, tuple) else load_pickle(case_properties)
        )
        logits = self._predict(case)
        logits, case_properties = reverse_preprocessing(
            crop_to_nonzero=self.plans["crop_to_nonzero"],
            images=logits,
            image_properties=case_properties,
            n_classes=self.plans["num_classes"],
            transpose_forward=self.plans["transpose_forward"],
            transpose_backward=self.plans["transpose_backward"],
        )
        return logits, case_properties

    def _predict(self, case):
        logits = self.model.predict(
            data=case,
            mode=self.model_dimensions,
            mirror=self.test_time_augmentation,
            overlap=self.sliding_window_overlap,
            patch_size=self.patch_size,
            sliding_window_prediction=self.sliding_window_prediction,
        )
        return logits

    def training_step(self, batch, batch_idx):  # noqa: U100
        inputs, target = batch["image"], batch["label"]
        output = self(inputs)
        loss = self.loss_fn_train(output, target)

        if self.deep_supervision:
            # If deep_supervision is enabled output and target will be a list of (downsampled) tensors.
            # We only need the original ground truth and its corresponding prediction which is always the first entry in each list.
            output = output[0]
            target = target[0]

        metrics = self.compute_metrics(self.train_metrics, output, target)
        self.log_dict(
            {"train/loss": loss} | metrics,
            on_step=self.step_logging,
            on_epoch=self.epoch_logging,
            prog_bar=self.progress_bar,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):  # noqa: U100
        inputs, target = batch["image"], batch["label"]
        output = self(inputs)
        loss = self.loss_fn_val(output, target)

        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            on_step=self.step_logging,
            on_epoch=self.epoch_logging,
            prog_bar=self.progress_bar,
            logger=True,
        )