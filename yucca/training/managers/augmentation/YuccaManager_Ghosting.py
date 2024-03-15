from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.augmentation.augmentation_presets import no_aug


class YuccaManager_Ghosting(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = no_aug
        self.augmentation_params["motion_ghosting_p_per_sample"] = 1.0
