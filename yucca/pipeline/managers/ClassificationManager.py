import torch
import torch.nn as nn
from yucca.pipeline.managers.YuccaManagerV2 import YuccaManagerV2, YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import genericV2
from yucca.modules.lightning_modules.ClassificationLightningModule import ClassificationLightningModule
from yucca.modules.data.datasets.ClassificationDataset import ClassificationDataset
from yucca.modules.optimization.loss_functions.CE import CE


class ClassificationManager(YuccaManagerV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = genericV2
        self.augmentation_params["skip_label"] = True
        self.model_name = "ResNet50_Volumetric"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationDataset


class ClassificationManagerV2(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = genericV2
        self.augmentation_params["skip_label"] = True
        self.model_name = "ResNet50_Volumetric"
        self.loss = CE
        self.lightning_module = ClassificationLightningModule
        self.model_dimensions = "3D"
        self.patch_based_training = False
        self.deep_supervision = False
        self.train_dataset_class = ClassificationDataset
