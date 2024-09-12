import torch
from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import genericV2


class YuccaManagerV2(YuccaManager):
    """
    Optimizations from the YuccaManager:

    - increased oversampling is shown to significantly improve performance on imbalanced datasets.
        self.p_oversample_foreground = 0.6

    - using generic augmentations rather than MR-specific ones
        self.augmentation_params = generic

    - enabled clipping to input range during augmentation
        self.augmentation_params["clip_to_input_range"] = True

    - adamw optimizer (tested LR: 1e-3, 5e-4, 1e-4)
        self.optimizer = torch.optim.AdamW
        self.optim_kwargs = {"eps": 1e-8, "betas": (0.9, 0.99), "lr": 1e-4, "weight_decay": 5e-2}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_supervision = True
        self.augmentation_params = genericV2
        self.optimizer = torch.optim.AdamW
        self.optim_kwargs = {"eps": 1e-8, "betas": (0.9, 0.99), "lr": 5e-5, "weight_decay": 5e-2}
        self.p_oversample_foreground = 0.6
