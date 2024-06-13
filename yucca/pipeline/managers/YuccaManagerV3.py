import torch
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import generic


class YuccaManagerV3(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = generic
        self.augmentation_params["clip_to_input_range"] = True
        self.learning_rate = 1e-2
        self.momentum = 0.99
        self.optimizer = torch.optim.AdamW
        self.optim_kwargs = {"eps": 1e-8, "betas": (0.9, 0.99), "lr": 5e-4, "weight_decay": 5e-2}
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
