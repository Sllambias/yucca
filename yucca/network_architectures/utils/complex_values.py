import torch

def Cmul(x, y): # Can perhaps be done with @-operator
    """
    Complex multiplication of two complex vectors.
    x: Tensor of shape [B, 2, C, X, Y, Z]
    y: Tensor of shape [B, 2, C, X, Y, Z]
    """
    a, b = x[:, 0], x[:, 1]
    c, d = y[:, 0], y[:, 1]

    real = a * c - b * d  # (a+ib)*(c+id)
    imag = b * c + a * d

    return torch.stack(
        [real, imag], dim=1
    )  # Concatenates a sequence of tensors along a new dimension. All tensors need to be of the same size.
