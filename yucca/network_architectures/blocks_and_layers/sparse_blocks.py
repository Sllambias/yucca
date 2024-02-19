import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional
from torch import Tensor
from src.models.conv_blocks.sparse_conv import SparseConv3d, SparseConv2d, SparseConvNeXtLayerNorm2d, SparseConvNeXtLayerNorm3d


class SparseMedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: int = True,
        n_groups: int or None = None,
        dim="3d",
        grn=False,
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ["2d", "3d"]
        self.dim = dim
        if self.dim == "2d":
            conv = SparseConv2d
            norm = SparseConvNeXtLayerNorm2d
        elif self.dim == "3d":
            conv = SparseConv3d
            norm = SparseConvNeXtLayerNorm3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        self.norm = norm(normalized_shape=in_channels)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # GeLU activations
        self.act = nn.GELU()

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.grn = grn
        if grn:
            if dim == "3d":
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1, 1), requires_grad=True)
            elif dim == "2d":
                self.grn_beta = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1, exp_r * in_channels, 1, 1), requires_grad=True)

    def forward(self, x, _dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == "3d":
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == "2d":
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class SparseMedNeXtDownBlock(SparseMedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            dim=dim,
            grn=grn,
        )

        if self.dim == "2d":
            conv = SparseConv2d
        elif self.dim == "3d":
            conv = SparseConv3d

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
            )

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, _dummy_tensor=None):
        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


# %%

# %%
