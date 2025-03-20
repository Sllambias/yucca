from typing import Optional
import torch
from torch import nn, Tensor


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        conv_op=nn.Conv2d,
        dropout_op: Optional[nn.Module] = None,
        dropout_kwargs={"p": 0.25},
        norm_op=nn.BatchNorm2d,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    ) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_k3(
            conv_op=conv_op, in_planes=inplanes, out_planes=planes, stride=stride, groups=groups, dilation=dilation
        )
        self.norm1 = norm_op(planes)
        self.relu = nonlin(**nonlin_kwargs)
        self.conv2 = conv_k3(conv_op=conv_op, in_planes=planes, out_planes=planes, stride=1, groups=1, dilation=1)
        self.norm2 = norm_op(planes)
        self.downsample = downsample
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_kwargs)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out


def conv_k1(conv_op, in_planes: int, out_planes: int, stride: int = 1):
    return conv_op(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_k3(conv_op, in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return conv_op(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        dropout_op: Optional[nn.Module] = None,
        dropout_kwargs={"p": 0.25},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_k1(conv_op=conv_op, in_planes=inplanes, out_planes=planes)
        self.bn1 = norm_op(width)
        self.conv2 = conv_k3(
            conv_op=conv_op, in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation
        )
        self.bn2 = norm_op(width)
        self.conv3 = conv_k1(conv_op=conv_op, in_planes=width, out_planes=planes * self.expansion)
        self.bn3 = norm_op(planes * self.expansion)
        self.relu = nonlin(**nonlin_kwargs)
        self.downsample = downsample
        self.stride = stride
        if dropout_op is not None:
            self.dropout = dropout_op(**dropout_kwargs)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        return out
