"""
Negative log likelihood loss wrapper
Makes loss function robust to target tensors with a channel dimension
"""

from torch import nn, Tensor


class CE(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
