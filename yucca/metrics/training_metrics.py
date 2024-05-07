from torch import Tensor
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy as TorchAccuracy
from torchmetrics.classification import AUROC as TorchAUROC
from torchmetrics import Metric


class F1(MulticlassF1Score):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target)


class AUROC(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.method = TorchAUROC(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return self.method(input, target.long())

    def compute(self):
        pass

    def update(self):
        pass


class Accuracy(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.method = TorchAccuracy(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return self.method(input, target.long())

    def compute(self):
        pass

    def update(self):
        pass
