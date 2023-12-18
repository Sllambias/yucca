from yucca.training.managers.YuccaManager import YuccaManager
from yucca.training.loss_and_optim.loss_functions.CE import CE


class YuccaManager_CE(YuccaManager):
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
    ):
        super().__init__(model, model_dimensions, task, folds, plan_id, starting_lr, loss_fn, momentum, continue_training)
        self._DEFAULT_LOSS = CE
