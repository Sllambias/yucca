from torch import Tensor, argmax, sigmoid
from torch import nn
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy as TorchAccuracy
from torchmetrics.classification import AUROC as TorchAUROC
from torchmetrics.segmentation import GeneralizedDiceScore as TorchGeneralizedDiceScore
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


class GeneralizedDiceScore(Metric):
    def __init__(self, multilabel: bool, num_classes: int, per_class: bool = False, average: bool = False, *args, **kwargs):
        super().__init__()
        self.multilabel = multilabel
        self.num_classes = num_classes

        self.per_class = per_class
        self.average = average

        self.method = TorchGeneralizedDiceScore(
            num_classes=self.num_classes, per_class=self.per_class or self.average, *args, **kwargs
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.multilabel:
            input = (sigmoid(input) > 0.5).long()
        else:
            # GeneralizedDice doesn't support logits, so we need to argmax it.
            input = argmax(input, dim=1)
            target = target.squeeze()

            # The following is due to a bug in torchmetrics. This can be removed when GeneralizedDiceScore has a property
            # argument `input_type` which is currently on their main branch, but released yet.
            # We should then set `input_type=one_hot` only when data is from regions.
            input = nn.functional.one_hot(input, num_classes=self.num_classes).movedim(-1, 1)
            target = nn.functional.one_hot(target.long(), num_classes=self.num_classes).movedim(-1, 1)
            assert len(input.shape) <= 5, input.shape
            assert len(target.shape) <= 5, target.shape

        output = self.method(input, target)

        if self.average:
            return output.mean()

        return output

    def compute(self):
        pass

    def update(self):
        pass
