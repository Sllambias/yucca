from yucca.pipeline.managers.YuccaManager import YuccaManager
from yucca.data.augmentation.augmentation_presets import generic


class YuccaManager_Julia(YuccaManager):
    def __init__(
        self,
        model,
        model_dimensions: str,
        task: str,
        folds: str | int,
        plan_id: str,
        starting_lr: float = None,
        loss_fn: str = None,
        momentum: float = None,
        continue_training: bool = False,
        augmentation_params: str = None,
    ):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, loss_fn, momentum, continue_training)
        self.loss = 'DC_SkelREC_and_CE_loss'
        self.augmentation_params = generic
        self.augmentation_params['skeleton'] = True
        


