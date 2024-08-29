import torch
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import generic


class YuccaManagerV4(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = generic
        self.augmentation_params["clip_to_input_range"] = True
        self.momentum = 0.99
        self.optimizer = torch.optim.AdamW
        self.optim_kwargs = {"eps": 1e-8, "betas": (0.9, 0.99), "lr": 1e-5, "weight_decay": 1e-3}
        self.p_oversample_foreground = 0.6
