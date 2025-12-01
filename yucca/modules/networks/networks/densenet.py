from __future__ import annotations
from collections import OrderedDict
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.densenet import _DenseBlock, _Transition


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        input_channels: number of the input channel.
        num_classes: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        conv_op: int,
        input_channels: int,
        num_classes: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(conv_op, nn.Conv2d):
            spatial_dims = 2
        else:
            spatial_dims = 3

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        self.avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(input_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        input_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=input_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            input_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=input_channels)
                )
            else:
                _out_channels = input_channels // 2
                trans = _Transition(spatial_dims, in_channels=input_channels, out_channels=_out_channels, act=act, norm=norm)
                self.features.add_module(f"transition{i + 1}", trans)
                input_channels = _out_channels

        # pooling and classification
        self.fc_channels = input_channels
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", self.avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(self.fc_channels, num_classes)),
                ]
            )
        )
        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        return x


class DenseNet121(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        conv_op,
        input_channels: int,
        num_classes: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        **kwargs,
    ) -> None:
        super().__init__(
            conv_op=conv_op,
            input_channels=input_channels,
            num_classes=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )


class DenseNet169(DenseNet):
    """DenseNet169 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        conv_op,
        input_channels: int,
        num_classes: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 32, 32),
        **kwargs,
    ) -> None:
        super().__init__(
            conv_op=conv_op,
            input_channels=input_channels,
            num_classes=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )


class DenseNet201(DenseNet):
    """DenseNet201 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        conv_op,
        input_channels: int,
        num_classes: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 48, 32),
        **kwargs,
    ) -> None:
        super().__init__(
            conv_op=conv_op,
            input_channels=input_channels,
            num_classes=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )


class DenseNet264(DenseNet):
    """DenseNet264"""

    def __init__(
        self,
        spatial_dims: int,
        input_channels: int,
        num_classes: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 64, 48),
        **kwargs,
    ) -> None:
        super().__init__(
            spatial_dims=spatial_dims,
            input_channels=input_channels,
            num_classes=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            **kwargs,
        )


class DenseNet_cov(DenseNet):
    def __init__(
        self,
        conv_op: int,
        input_channels: int,
        num_classes: int,
        n_covariates: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__(
            conv_op=conv_op,
            input_channels=input_channels,
            num_classes=num_classes,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            act=act,
            norm=norm,
            dropout_prob=dropout_prob,
        )

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", self.avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("out", nn.Linear(self.fc_channels, 100 - n_covariates)),
                ]
            )
        )
        self.out = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers(x)
        x = torch.concat((x, cov.to(x.dtype)), dim=1)
        x = self.out(x)
        return x


def densenet121_2cov(
    conv_op,
    input_channels: int,
    num_classes: int,
    init_features: int = 64,
    growth_rate: int = 32,
    block_config: Sequence[int] = (6, 12, 24, 16),
):
    return DenseNet_cov(
        conv_op=conv_op,
        input_channels=input_channels,
        num_classes=num_classes,
        n_covariates=2,
        init_features=init_features,
        growth_rate=growth_rate,
        block_config=block_config,
    )


Densenet = DenseNet
Densenet121 = densenet121 = DenseNet121
Densenet169 = densenet169 = DenseNet169
Densenet201 = densenet201 = DenseNet201
Densenet264 = densenet264 = DenseNet264

if __name__ == "__main__":
    im = torch.ones((2, 1, 32, 32, 32))
    cov = torch.ones(2, 3)
    net = DenseNet_cov(conv_op=torch.nn.Conv3d, n_covariates=3, input_channels=1, num_classes=4)
    out = net(im, cov)
    print(out)
