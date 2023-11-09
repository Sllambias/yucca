from yucca.training.trainers.YuccaManager import YuccaManager
from torch import optim


class YuccaManager_AdamW(YuccaManager):
    """
    The difference from YuccaManagerV2 --> YuccaManagerV3 is:
    - Introduces Deep Supervision
    - Uses data augmentation scheme V2
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = optim.AdamW
