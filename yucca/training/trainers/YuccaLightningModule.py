#%%
from typing import Any, Optional
import lightning as L
import torch
import yuccalib
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.preprocessing.YuccaPreprocessor import YuccaPreprocessor
from yucca.paths import yucca_preprocessed
from yuccalib.utils.files_and_folders import recursive_find_python_class
from yuccalib.loss_and_optim.loss_functions.CE import CE
from yuccalib.utils.kwargs import filter_kwargs


class YuccaLightningModule(L.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-3,
            loss_fn: nn.Module = CE,
            model_name: str = 'UNet',
            model_dimensions: str = '3D',
            momentum: float = 0.9,
            num_modalities: int = 1,
            num_classes: int = 1,
            optimizer: torch.optim.Optimizer = torch.optim.SGD,
            patch_size: list | tuple = None,
            plans_path: str = None,
            sliding_window_overlap: float = 0.5,
            test_time_augmentation: bool = False
            ):
        super().__init__()
        # Model parameters
        self.model_name = model_name
        self.model_dimensions = model_dimensions
        self.num_classes = num_classes
        self.num_modalities = num_modalities

        # Loss, optimizer and scheduler parameters
        self.lr = learning_rate
        self.loss_fn = loss_fn
        self.momentum = momentum
        self.optim = optimizer

        # Inference
        self.sliding_window_overlap = sliding_window_overlap
        self.patch_size = patch_size
        self.plans_path = plans_path
        self.test_time_augmentation = test_time_augmentation

        # Save params and start training
        self.save_hyperparameters()
        self.initialize()

    def initialize(self):
        self.model = recursive_find_python_class(
            folder=[join(yuccalib.__path__[0], 'network_architectures')],
            class_name=self.model_name,
            current_module='yuccalib.network_architectures')

        if self.model_dimensions == '3D':
            conv_op = torch.nn.Conv3d
            norm_op = torch.nn.InstanceNorm3d

        self.model = self.model(
            input_channels=self.num_modalities, 
            conv_op=conv_op, 
            norm_op=norm_op, 
            num_classes=self.num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch['image'], batch['seg']
        output = self(inputs.float())
        loss = self.loss_fn(output.softmax(1), target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch['image'], batch['seg']
        output = self(inputs.float())
        loss = self.loss_fn(output.softmax(1), target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_predict_start(self):
        self.preprocessor = YuccaPreprocessor(self.plans_path)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        case, case_id = batch
        case_preprocessed, case_properties = self.preprocessor.preprocess_case_for_inference(case, self.patch_size)
        logits = self.model.predict(mode=self.model_dimensions,
                                    data=case_preprocessed,
                                    patch_size=self.patch_size,
                                    overlap=self.sliding_window_overlap,
                                    mirror=self.test_time_augmentation).detach().cpu().numpy()
        logits = self.preprocessor.reverse_preprocessing(logits, case_properties)
        return {'logits': logits, 'properties': case_properties, 'case_id': case_id[0]}

    def configure_optimizers(self):
        # Initialize the loss function using relevant kwargs
        loss_kwargs = {}
        loss_kwargs = filter_kwargs(self.loss_fn, loss_kwargs)
        self.loss_fn = self.loss_fn(**loss_kwargs)

        # optim_kwargs holds arguments for all possible optimizers we could use.
        # The subsequent filter_kwargs will select only the kwargs that can be passed 
        # to the chosen optimizer
        optim_kwargs = {
            'lr': self.lr,
            'momentum': self.momentum}
        optim_kwargs = filter_kwargs(self.optim, optim_kwargs)
        return self.optim(self.model.parameters(), lr=self.lr)
    

# %%
