"""
Negative log likelihood loss wrapper
Makes loss function robust to target tensors with a channel dimension
"""

import torch
from torch import nn, Tensor


class NLL(nn.NLLLoss):
    def __init__(self, log=False):
        super().__init__()
        self.log = log

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        if self.log:
            return super().forward(torch.log(input), target)
        return super().forward(input, target)
