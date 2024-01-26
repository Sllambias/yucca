"""
FROM: https://github.com/j-sripad/mulitresunet-pytorch/blob/main/multiresunet.py
"""

import torch
import torch.nn as nn
from yucca.network_architectures.networks.YuccaNet import YuccaNet
from yucca.network_architectures.blocks_and_layers.conv_blocks import (
    Multiresblock,
    Respath,
)
from yucca.network_architectures.blocks_and_layers.conv_layers import (
    ConvDropoutNormNonlin,
)


class MultiResUNet(YuccaNet):
    """
    Arguments:
    channels - input image channels
    filters - filters to begin with (Unet)
    nclasses - number of classes

    Returns - None
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        starting_filters: int = 32,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        dropout_in_decoder=False,
        weightInitializer=None,
        basic_block=ConvDropoutNormNonlin,
    ) -> None:
        super(MultiResUNet, self).__init__()
        # Architecture specific
        self.alpha = 1.67

        # Task specific
        self.num_classes = num_classes
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block

        # Dimensions
        if self.conv_op == nn.Conv2d:
            self.pool_op = torch.nn.MaxPool2d
            self.upsample = torch.nn.ConvTranspose2d
        else:
            self.pool_op = torch.nn.MaxPool3d
            self.upsample = torch.nn.ConvTranspose3d

        # Encoder Path
        self.multiresblock1 = Multiresblock(
            input_channels,
            self.filters,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters1 = (
            int(self.filters * self.alpha * 0.167)
            + int(self.filters * self.alpha * 0.333)
            + int(self.filters * self.alpha * 0.5)
        )
        self.pool1 = self.pool_op(2)
        self.respath1 = Respath(
            self.in_filters1,
            self.filters,
            4,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.multiresblock2 = Multiresblock(
            self.in_filters1,
            self.filters * 2,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters2 = (
            int(self.filters * 2 * self.alpha * 0.167)
            + int(self.filters * 2 * self.alpha * 0.333)
            + int(self.filters * 2 * self.alpha * 0.5)
        )
        self.pool2 = self.pool_op(2)
        self.respath2 = Respath(
            self.in_filters2,
            self.filters * 2,
            3,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.multiresblock3 = Multiresblock(
            self.in_filters2,
            self.filters * 4,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters3 = (
            int(self.filters * 4 * self.alpha * 0.167)
            + int(self.filters * 4 * self.alpha * 0.333)
            + int(self.filters * 4 * self.alpha * 0.5)
        )
        self.pool3 = self.pool_op(2)
        self.respath3 = Respath(
            self.in_filters3,
            self.filters * 4,
            2,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.multiresblock4 = Multiresblock(
            self.in_filters3,
            self.filters * 8,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters4 = (
            int(self.filters * 8 * self.alpha * 0.167)
            + int(self.filters * 8 * self.alpha * 0.333)
            + int(self.filters * 8 * self.alpha * 0.5)
        )
        self.pool4 = self.pool_op(2)
        self.respath4 = Respath(
            self.in_filters4,
            self.filters * 8,
            1,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.multiresblock5 = Multiresblock(
            self.in_filters4,
            self.filters * 16,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        if not dropout_in_decoder:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        # Decoder path
        self.in_filters5 = (
            int(self.filters * 16 * self.alpha * 0.167)
            + int(self.filters * 16 * self.alpha * 0.333)
            + int(self.filters * 16 * self.alpha * 0.5)
        )
        self.upsample6 = self.upsample(self.in_filters5, self.filters * 8, kernel_size=(2), stride=(2))

        self.multiresblock6 = Multiresblock(
            self.filters * 8 * 2,
            self.filters * 8,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters6 = (
            int(self.filters * 8 * self.alpha * 0.167)
            + int(self.filters * 8 * self.alpha * 0.333)
            + int(self.filters * 8 * self.alpha * 0.5)
        )
        self.upsample7 = self.upsample(self.in_filters6, self.filters * 4, kernel_size=(2), stride=(2))

        self.multiresblock7 = Multiresblock(
            self.filters * 4 * 2,
            self.filters * 4,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters7 = (
            int(self.filters * 4 * self.alpha * 0.167)
            + int(self.filters * 4 * self.alpha * 0.333)
            + int(self.filters * 4 * self.alpha * 0.5)
        )
        self.upsample8 = self.upsample(self.in_filters7, self.filters * 2, kernel_size=(2), stride=(2))

        self.multiresblock8 = Multiresblock(
            self.filters * 2 * 2,
            self.filters * 2,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters8 = (
            int(self.filters * 2 * self.alpha * 0.167)
            + int(self.filters * 2 * self.alpha * 0.333)
            + int(self.filters * 2 * self.alpha * 0.5)
        )
        self.upsample9 = self.upsample(self.in_filters8, self.filters, kernel_size=(2), stride=(2))

        self.multiresblock9 = Multiresblock(
            self.filters * 2,
            self.filters,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
            self.basic_block,
        )

        self.in_filters9 = (
            int(self.filters * self.alpha * 0.167)
            + int(self.filters * self.alpha * 0.333)
            + int(self.filters * self.alpha * 0.5)
        )
        self.conv_final = self.conv_op(self.in_filters9, self.num_classes, 1, 1, 0, 1, 1, False)

        if not dropout_in_decoder:
            self.dropout_op_kwargs["p"] = old_dropout_p

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)
        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        logits = self.conv_final(x_multires9)

        return logits
