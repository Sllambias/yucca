import torch
import numpy as np
from yucca.training.trainers.YuccaTrainer import YuccaTrainer
from torch import optim


class YuccaTrainer_AdamW(YuccaTrainer):
    """
    The difference from YuccaTrainerV2 --> YuccaTrainerV3 is:
    - Introduces Deep Supervision
    - Uses data augmentation scheme V2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optim = optim.AdamW
