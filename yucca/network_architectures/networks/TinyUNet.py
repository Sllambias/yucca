import torch
import torch.nn as nn
from yucca.network_architectures.networks.YuccaNet import YuccaNet
from yucca.network_architectures.blocks_and_layers.conv_layers import (
    DoubleConvDropoutNormNonlin,
)


class TinyUNet(YuccaNet):
    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        starting_filters: int = 4,
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
        basic_block=DoubleConvDropoutNormNonlin,
        deep_supervision=False,
    ) -> None:
        super(TinyUNet, self).__init__()

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
        self.deep_supervision = deep_supervision

        # Dimensions
        if self.conv_op == nn.Conv2d:
            self.pool_op = torch.nn.MaxPool2d
            self.upsample = torch.nn.ConvTranspose2d
        else:
            self.pool_op = torch.nn.MaxPool3d
            self.upsample = torch.nn.ConvTranspose3d

        self.in_conv = self.basic_block(
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
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            self.filters,
            self.filters * 2,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            self.filters * 2,
            self.filters * 4,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        # Decoder
        if not dropout_in_decoder:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        self.upsample1 = self.upsample(self.filters * 4, self.filters * 2, kernel_size=2, stride=2)
        self.decoder_conv1 = self.basic_block(
            self.filters * 4,
            self.filters * 2,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(self.filters * 2, self.filters, kernel_size=2, stride=2)
        self.decoder_conv2 = self.basic_block(
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
        )

        self.out_conv = self.conv_op(self.filters, self.num_classes, kernel_size=1)

        if not dropout_in_decoder:
            self.dropout_op_kwargs["p"] = old_dropout_p

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        # Encoder path
        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        # Decoder path
        x3 = torch.cat([self.upsample1(x2), x1], dim=1)
        x3 = self.decoder_conv1(x3)

        x4 = torch.cat([self.upsample2(x3), x0], dim=1)
        x4 = self.decoder_conv2(x4)

        if self.deep_supervision:
            pass

        logits = self.out_conv(x4)
        return logits
