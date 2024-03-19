import lightning as L
import torch
import yucca
import wandb
import copy
import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from batchgenerators.utilities.file_and_folder_operations import join
from torchmetrics import MetricCollection
from torchmetrics.classification import Dice, Accuracy, AUROC
from torchmetrics.regression import MeanAbsoluteError
from yucca.training.loss_and_optim.loss_functions.deep_supervision import DeepSupervisionLoss
from yucca.utils.files_and_folders import recursive_find_python_class
from yucca.utils.kwargs import filter_kwargs
from yucca.evaluation.training_metrics import F1


class YuccaLightningModule(L.LightningModule):
    """
    The YuccaLightningModule class is an implementation of the PyTorch Lightning module designed for the Yucca project.
    It extends the LightningModule class and encapsulates the neural network model, loss functions, and optimization logic.
    This class is responsible for handling training, validation, and inference steps within the Yucca machine learning pipeline.


    """

    def __init__(
        self,
        config: dict,
        deep_supervision: bool = False,
        learning_rate: float = 1e-3,
        loss_fn: str = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR,
        momentum: float = 0.9,
        optimizer: torch.optim.Optimizer = torch.optim.SGD,
        sliding_window_overlap: float = 0.5,
        step_logging: bool = False,
        test_time_augmentation: bool = False,
        progress_bar: bool = False,
    ):
        super().__init__()
        # Extract parameters from the configurator
        self.num_classes = config["num_classes"]
        self.num_modalities = config["num_modalities"]
        self.version_dir = config["version_dir"]
        self.plans = config["plans"]
        self.plans_path = config["plans_path"]
        self.model_name = config["model_name"]
        self.model_dimensions = config["model_dimensions"]
        self.patch_size = config["patch_size"]
        self.task_type = config["task_type"]
        self.sliding_window_prediction = config["patch_based_training"]

        # Loss, optimizer and scheduler parameters
        self.deep_supervision = deep_supervision
        self.lr = learning_rate
        self.loss_fn = loss_fn

        self.momentum = momentum
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        # Evaluation and logging
        if step_logging is True:  # Blame PyTorch lightning for this war crime
            self.step_logging = None
            self.epoch_logging = None
        else:
            self.step_logging = False
            self.epoch_logging = True

        self.progress_bar = progress_bar

        if self.task_type == "classification":
            tmetrics_task = "multiclass" if self.num_classes > 2 else "binary"
            # can we get per-class?
            self.train_metrics = MetricCollection(
                {
                    "train_acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                    "train_roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
                }
            )
            self.val_metrics = MetricCollection(
                {
                    "val_acc": Accuracy(task=tmetrics_task, num_classes=self.num_classes),
                    "val_roc_auc": AUROC(task=tmetrics_task, num_classes=self.num_classes),
                }
            )

        if self.task_type == "segmentation":
            self.train_metrics = MetricCollection(
                {
                    "train/dice": Dice(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None),
                    "train/F1": F1(
                        num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None, average=None
                    ),
                },
            )

            self.val_metrics = MetricCollection(
                {
                    "val/dice": Dice(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None),
                    "val/F1": F1(num_classes=self.num_classes, ignore_index=0 if self.num_classes > 1 else None, average=None),
                },
            )
            _default_loss = "DiceCE"

        if self.task_type == "self-supervised":
            self.train_metrics = MetricCollection({"train/MAE": MeanAbsoluteError()})
            self.val_metrics = MetricCollection({"train/MAE": MeanAbsoluteError()})
            _default_loss = "MSE"

        if self.loss_fn is None:
            self.loss_fn = _default_loss

        self.log_image_every_n_epochs = 1
        # Inference
        self.sliding_window_overlap = sliding_window_overlap
        self.test_time_augmentation = test_time_augmentation

        # If we are training we save params and then start training
        # Do not overwrite parameters during inference.
        self.save_hyperparameters()
        self.load_model()

    def load_model(self):
        logging.info(f"Loading Model: {self.model_dimensions} {self.model_name}")
        self.model = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "network_architectures")],
            class_name=self.model_name,
            current_module="yucca.network_architectures",
        )
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

    def forward(self, inputs):
        return self.model(inputs)

    def teardown(self, stage: str):  # noqa: U100
        wandb.finish()

    def training_step(self, batch, batch_idx):
        inputs, target, file_path = batch["image"], batch["label"], batch["file_path"]
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

        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0 and wandb.run is not None:
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
        metrics = self.compute_metrics(self.val_metrics, output, target)
        self.log_dict(
            {"val/loss": loss} | metrics,
            on_step=self.step_logging,
            on_epoch=self.epoch_logging,
            prog_bar=self.progress_bar,
            logger=True,
        )

        if batch_idx == 0 and self.current_epoch % self.log_image_every_n_epochs == 0 and wandb.run is not None:
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

    def on_predict_start(self):
        preprocessor_class = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "preprocessing")],
            class_name=self.plans["preprocessor"],
            current_module="yucca.preprocessing",
        )
        self.preprocessor = preprocessor_class(join(self.version_dir, "hparams.yaml"))

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        case, case_id = batch
        (
            case_preprocessed,
            case_properties,
        ) = self.preprocessor.preprocess_case_for_inference(case, self.patch_size, self.sliding_window_prediction)
        logits = self.model.predict(
            data=case_preprocessed,
            mode=self.model_dimensions,
            mirror=self.test_time_augmentation,
            overlap=self.sliding_window_overlap,
            patch_size=self.patch_size,
            sliding_window_prediction=self.sliding_window_prediction,
        )

        logits, case_properties = self.preprocessor.reverse_preprocessing(logits, case_properties)
        return {"logits": logits, "properties": case_properties, "case_id": case_id[0]}

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
        # Initialize and configure the loss(es) here.
        # loss_kwargs holds args for any scheduler class,
        # but using filtering we only pass arguments relevant to the selected class.
        self.loss_fn = recursive_find_python_class(
            folder=[join(yucca.__path__[0], "training/loss_and_optim")],
            class_name=self.loss_fn,
            current_module="yucca.training.loss_and_optim",
        )
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

    def _log_dict_of_images_to_wandb(self, imagedict: {}, log_key: str, task_type: str = "segmentation"):
        # This needs to handle the following cases:
        # Segmentation      : {"input": (b,m,x,y(,z)), "target": (b,1,x,y(,z)), "output": (b,c,x,y(,z)), "file_path": [pathA, pathB, ...]}
        # Self-supervised   : {"input": (b,m,x,y(,z)), "target": (b,m,x,y(,z)), "output": (b,m,x,y(,z)), "file_path": [pathA, pathB, ...]}
        # Classification    : {"input": (b,m,x,y(,z)), "target": (b,1,x), "output": (b,c,x), "file_path": [pathA, pathB, ...]}

        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        channel_idx = np.random.randint(0, imagedict["input"].shape[1])

        if len(imagedict["input"].shape) == 5:  # 3D images.
            # We need to select a slice to visualize.
            if task_type == "segmentation" and len(imagedict["target"][batch_idx, 0].nonzero()[0]) > 0:
                # Select a foreground slice if any exist.
                foreground_locations = imagedict["target"][batch_idx, 0].nonzero()
                slice_to_visualize = foreground_locations[0][np.random.randint(0, len(foreground_locations[0]))]
            else:
                slice_to_visualize = np.random.randint(0, imagedict["input"].shape[2])

            imagedict["input"] = imagedict["input"][:, :, slice_to_visualize]
            if len(imagedict["target"].shape) == 5:
                imagedict["target"] = imagedict["target"][:, :, slice_to_visualize]
            if len(imagedict["output"].shape) == 5:
                imagedict["output"] = imagedict["output"][:, :, slice_to_visualize]

        image = imagedict["input"][batch_idx, channel_idx]
        case = os.path.splitext(os.path.split(imagedict["file_path"][batch_idx])[-1])[0]

        if task_type in ["segmentation", "classification"]:
            target = imagedict["target"][batch_idx, 0]
            output = imagedict["output"][batch_idx].argmax(0)
        elif task_type == "self-supervised":
            target = imagedict["target"][batch_idx, channel_idx]
            output = imagedict["output"][batch_idx, channel_idx]
        else:
            logging.warn(
                f"Unknown task type. Found {task_type} and expected one in ['classification', 'segmentation', 'self-supervised']"
            )

        if len(target.shape) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=100, constrained_layout=True)
            axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
            axes[0].set_title("input")
            fig.suptitle(f"{case}. Target: {target} | Output: {output}", fontsize=16)
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), dpi=100, constrained_layout=True)
            axes[0].imshow(image, cmap="gray", vmin=np.quantile(image, 0.01), vmax=np.quantile(image, 0.99))
            axes[0].set_title("input")
            axes[1].imshow(target, cmap="gray")
            axes[1].set_title("target")
            axes[2].imshow(output, cmap="gray")
            axes[2].set_title("output")
            fig.suptitle(case, fontsize=16)

            wandb.log({log_key: wandb.Image(fig)}, commit=False)
            plt.close(fig)
