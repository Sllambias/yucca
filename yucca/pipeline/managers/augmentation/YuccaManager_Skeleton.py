from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.modules.data.augmentation.augmentation_presets import generic
from yucca.modules.optimization.loss_functions.Skeleton_recall_loss import SoftSkeletonRecallLoss


class YuccaManager_Skeleton(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = generic
        self.loss = SoftSkeletonRecallLoss
