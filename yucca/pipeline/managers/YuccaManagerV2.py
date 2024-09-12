import torch
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import genericV2


class YuccaManagerV2(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = genericV2
        self.optimizer = torch.optim.AdamW
        self.optim_kwargs = {"eps": 1e-8, "betas": (0.9, 0.99), "lr": 5e-5, "weight_decay": 5e-2}
        self.p_oversample_foreground = 0.6
