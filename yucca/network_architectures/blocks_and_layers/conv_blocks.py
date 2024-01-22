import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yucca.network_architectures.blocks_and_layers.conv_layers import (
    ConvDropoutNormNonlin,
    ConvDropoutNorm,
)
from yucca.network_architectures.blocks_and_layers.norm import LayerNorm3d
from timm.layers import DropPath
from functools import partial
from typing import Callable, Optional


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        return out


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: int = True,
        norm_type: str = "group",
        n_groups: int or None = None,
        dim="3d",
        grn=False,
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ["2d", "3d"]
        self.dim = dim
        if self.dim == "2d":
            conv = nn.Conv2d
        elif self.dim == "3d":
            conv = nn.Conv3d

        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(normalized_shape=in_channels, data_format="channels_first")

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


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        if dim == "2d":
            conv = nn.Conv2d
        elif dim == "3d":
            conv = nn.Conv3d
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


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_r=4,
        kernel_size=7,
        do_res=False,
        norm_type="group",
        dim="3d",
        grn=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.resample_do_res = do_res

        self.dim = dim
        if dim == "2d":
            conv = nn.ConvTranspose2d
        elif dim == "3d":
            conv = nn.ConvTranspose3d
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
        # Asymmetry but necessary to match shape

        if self.dim == "2d":
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        elif self.dim == "3d":
            x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == "2d":
                res = torch.nn.functional.pad(res, (1, 0, 1, 0))
            elif self.dim == "3d":
                res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):
    def __init__(self, in_channels, n_classes, dim):
        super().__init__()

        if dim == "2d":
            conv = nn.ConvTranspose2d
        elif dim == "3d":
            conv = nn.ConvTranspose3d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, _dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, _dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class Multiresblock(nn.Module):
    """
    MultiRes Block

    Arguments:
    num_in_channels {int} -- Number of channels coming into mutlires block
    num_filters {int} -- Number of filters in a corrsponding UNet stage
    alpha {float} -- alpha hyperparameter (default: 1.67)
    """

    def __init__(
        self,
        num_in_channels,
        num_filters,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        basic_block=ConvDropoutNormNonlin,
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.alpha = 1.67
        self.W = num_filters * self.alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        shortcut_kwargs = {"kernel_size": 1}

        self.shortcut = ConvDropoutNorm(
            num_in_channels,
            num_out_filters,
            self.conv_op,
            shortcut_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.conv_3x3 = basic_block(
            num_in_channels,
            filt_cnt_3x3,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.conv_5x5 = basic_block(
            filt_cnt_3x3,
            filt_cnt_5x5,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.conv_7x7 = basic_block(
            filt_cnt_5x5,
            filt_cnt_7x7,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.norm1 = self.norm_op(num_out_filters, **self.norm_op_kwargs)
        self.norm2 = self.norm_op(num_out_filters, **self.norm_op_kwargs)
        self.activation = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.norm1(x)

        x = x + shrtct
        x = self.norm2(x)
        x = self.activation(x)

        return x


class Respath(torch.nn.Module):
    """
    ResPath

    Arguments:
    num_in_filters {int} -- Number of filters going in the respath
    num_out_filters {int} -- Number of filters going out the respath
    respath_length {int} -- length of ResPath

    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        respath_length,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        basic_block=ConvDropoutNormNonlin,
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        shortcut_kwargs = {"kernel_size": 1}

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.activation = self.nonlin(**self.nonlin_kwargs)

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    ConvDropoutNorm(
                        num_in_filters,
                        num_out_filters,
                        self.conv_op,
                        shortcut_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                )
                self.convs.append(
                    basic_block(
                        num_in_filters,
                        num_out_filters,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                )

            else:
                self.shortcuts.append(
                    ConvDropoutNorm(
                        num_out_filters,
                        num_out_filters,
                        self.conv_op,
                        shortcut_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                )

                self.convs.append(
                    basic_block(
                        num_out_filters,
                        num_out_filters,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    )
                )

            self.bns.append(self.norm_op(num_out_filters, **self.norm_op_kwargs))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.activation(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = self.activation(x)
        return x


class ux_block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        conv_op=nn.Conv3d,
        norm_op=LayerNorm3d,
        norm_op_kwargs={"eps": 1e-6},
        nonlin=nn.GELU,
    ):
        super().__init__()
        if conv_op == nn.Conv2d:
            self.permute_fw = [0, 2, 3, 1]
            self.permute_bw = [0, 3, 1, 2]
        else:
            self.permute_fw = [0, 2, 3, 4, 1]
            self.permute_bw = [0, 4, 1, 2, 3]

        self.dwconv = conv_op(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = norm_op(dim, **norm_op_kwargs)
        self.pwconv1 = conv_op(dim, 4 * dim, kernel_size=1, groups=dim)
        self.nonlin = nonlin()
        self.pwconv2 = conv_op(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.nonlin(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = x.permute(self.permute_fw)
            x = self.gamma * x
            x = x.permute(self.permute_bw)

        x = input + self.drop_path(x)
        return x


class UXNet_encoder(nn.Module):
    """
    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_channels=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=LayerNorm3d,
        norm_op_kwargs={"eps": 1e-6},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.GELU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        basic_block=ux_block,
    ):
        super().__init__()

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.basic_block = basic_block

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            self.conv_op(in_channels, dims[0], kernel_size=7, stride=2, padding=3),
            self.norm_op(dims[0], **self.norm_op_kwargs),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                self.norm_op(dims[i], **self.norm_op_kwargs),
                self.conv_op(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    self.basic_block(
                        dim=dims[i],
                        conv_op=self.conv_op,
                        norm_op=self.norm_op,
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(self.norm_op, eps=self.norm_op_kwargs["eps"])
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # self.apply(self._init_weights)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x
