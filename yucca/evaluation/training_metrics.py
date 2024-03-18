from torchmetrics.classification import MulticlassF1Score
from torch import Tensor


class F1(MulticlassF1Score):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target)
