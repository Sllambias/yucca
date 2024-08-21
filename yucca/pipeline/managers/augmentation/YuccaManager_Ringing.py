from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import no_aug


class YuccaManager_Ringing(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = no_aug
        self.augmentation_params["gibbs_ringing_p_per_sample"] = 1.0
