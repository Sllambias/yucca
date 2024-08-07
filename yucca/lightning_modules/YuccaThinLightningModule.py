import torch
import logging
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice
from yucca.pipeline.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.optimization.loss_functions.deep_supervision import DeepSupervisionLoss
from yucca.functional.utils.kwargs import filter_kwargs
from yucca.metrics.training_metrics import F1
from yucca.optimization.loss_functions.nnUNet_losses import DiceCE
from yucca.lightning_modules.YuccaLightningModule import YuccaLightningModule


class YuccaThinLightningModule(YuccaLightningModule):
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
        learning_rate: float = 1e-3,
        loss_fn: torch.nn.Module = DiceCE,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
        momentum: float = 0.9,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        optimizer_args: dict = {},  # WIP
        patch_based_training: bool = True,
        preprocessor: YuccaPreprocessor = None,
        sliding_window_overlap: float = 0.5,
        step_logging: bool = False,
        test_time_augmentation: bool = False,
        progress_bar: bool = False,
        log_image_every_n_epochs: int = None,  # FatVersion
    ):
        super().__init__()
        # Extract parameters from the configurator
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.hparams_path = hparams_path
        self.plans = plans
        self.model = model
        self.model_dimensions = model_dimensions
        self.patch_size = patch_size
        self.preprocessor = preprocessor
        self.sliding_window_prediction = patch_based_training

        # Loss, optimizer and scheduler parameters
        self.deep_supervision = deep_supervision
        self.disable_inference_preprocessing = disable_inference_preprocessing
        self.lr = learning_rate
        self.loss_fn = loss_fn
        self.momentum = momentum
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

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

        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.progress_bar = progress_bar

        logging.info(f"{self.__class__.__name__} initialized")
        logging.info(f"Deep Supervision Enabled: {self.deep_supervision}")

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

        # If we are training we save params and then start training
        # Do not overwrite parameters during inference.
        self.save_hyperparameters()
        self.load_model()

    def load_model(self):
        logging.info(f"Loading Model: {self.model_dimensions} {self.model.__class__.__name__}")
        if self.model_dimensions == "3D":
            conv_op = torch.nn.Conv3d
            norm_op = torch.nn.InstanceNorm3d
        else:
            conv_op = torch.nn.Conv2d
            norm_op = torch.nn.BatchNorm2d

        model_kwargs = {
            # Applies to all models
            "input_channels": self.num_modalities,
            "num_classes": self.num_classes,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            "deep_supervision": self.deep_supervision,
            "norm_op": norm_op,
            # UNetR
            "patch_size": self.patch_size,
            # MedNeXt
            "checkpoint_style": "outside_block",
        }
        model_kwargs = filter_kwargs(self.model, model_kwargs)
        self.model = self.model(**model_kwargs)

    def on_predict_start(self):
        if self.disable_inference_preprocessing:
            self.predict = self.predict_without_preprocessing
        else:
            self.preprocessor = self.preprocessor(self.hparams_path)
            self.predict = self.predict_with_preprocessing

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        case, case_id = batch
        logits, case_properties = self.predict(case)
        return {"logits": logits, "properties": case_properties, "case_id": case_id[0]}

    def configure_optimizers(self):
        # Initialize and configure the loss(es) here.
        # loss_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        loss_kwargs = {
            # DCE
            "soft_dice_kwargs": {"apply_softmax": True},
        }

        loss_kwargs = filter_kwargs(self.loss_fn, loss_kwargs)

        self.loss_fn_train = self.loss_fn(**loss_kwargs)
        self.loss_fn_val = self.loss_fn(**loss_kwargs)

        # If deep_supervision is enabled we wrap our training loss (and potentially specify weights)
        # We leave the validation loss as is, as deep_supervision is not used for validation.
        if self.deep_supervision:
            self.loss_fn_train = DeepSupervisionLoss(self.loss_fn_train, weights=None)

        # Initialize and configure the optimizer(s) here.
        # optim_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        optim_kwargs = {
            # all
            "lr": self.lr,
            # SGD
            "momentum": self.momentum,
            "eps": 1e-4,
            "weight_decay": 3e-5,
        }

        optim_kwargs = filter_kwargs(self.optim, optim_kwargs)

        self.optim = self.optim(self.model.parameters(), **optim_kwargs)

        # Initialize and configure LR scheduler(s) here
        # lr_scheduler_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        lr_scheduler_kwargs = {
            # Cosine Annealing
            "T_max": self.trainer.max_epochs,
            "eta_min": 1e-9,
        }

        lr_scheduler_kwargs = filter_kwargs(self.lr_scheduler, lr_scheduler_kwargs)

        self.lr_scheduler = self.lr_scheduler(self.optim, **lr_scheduler_kwargs)

        # Finally return the optimizer and scheduler - the loss is not returned.
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}
