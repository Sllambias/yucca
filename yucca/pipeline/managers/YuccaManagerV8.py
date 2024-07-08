import torch
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import generic


class YuccaManagerV8(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = generic
        self.momentum = 0.99
        self.p_oversample_foreground = 0.6
