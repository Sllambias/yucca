"""
Mean Squared Error loss wrapper
Makes loss function robust to target tensors with a channel dimension
"""

from torch import nn, Tensor


class MSE(nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)
