from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import skeleton
from yucca.optimization.loss_functions.Skeleton_recall_loss import SoftSkeletonRecallLoss



class YuccaManager_Skeleton(YuccaManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augmentation_params = skeleton
        self.loss = SoftSkeletonRecallLoss

