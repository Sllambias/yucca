# Original filename; CUNet_conjlayer_after_first_conv_with_bn.py
##############################################################################################
#                                                                                            #
#     Minor modifications of the original code, coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: This code implements U-Net with batch normalization in both complex and   #
#     real modes. In the complex mode architecture, a conjugate layer is strategically       #
#     positioned after the first convolution layer. It is important to note that             #
#     theoretically, this network is phase invariant.                                        #
#                                                                                            #
##############################################################################################

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from yucca.network_architectures.blocks_and_layers.complex_layers import (
    ComplexConv3d,
    ComplexConvFast3d,
    ComplexConvTranspose3d,
    NaiveCBN,
    GTReLU,
    ConjugateLayer,
)


class ConvNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels, stride, mode, conv_type, depth):
        super(ConvNormNonlin, self).__init__()

        self.mode = mode

        if self.mode == "complex":
            self.nonlin = GTReLU
            self.norm_op = NaiveCBN
            self.inv_layer = ConjugateLayer(output_channels, 3, conv_type)
            if conv_type == "fast":
                self.conv_op = ComplexConvFast3d
            else:
                self.conv_op = ComplexConv3d

        elif self.mode == "real":
            self.nonlin = nn.LeakyReLU
            self.norm_op = nn.BatchNorm3d
            self.conv_op = nn.Conv3d
        self.depth = depth

        self.conv = self.conv_op(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1
        )
        self.instnorm = self.norm_op(
            output_channels, eps=1e-5, momentum=0.1, affine=True
        )
        if self.mode == "real":
            self.lrelu = self.nonlin(negative_slope=1e-2, inplace=True)
        elif self.mode == "complex":
            self.lrelu = self.nonlin(output_channels, negative_slope=1e-2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.depth == 0 and self.mode == "complex":
            x = self.inv_layer(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(
        self,
        input_feature_channels,
        output_feature_channels,
        num_convs,
        first_stride,
        mode,
        conv_type,
        depth,
    ):
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels
        self.kernel_size = 3
        self.pad = 1
        self.mode = mode
        self.conv_type = conv_type
        self.first_stride = first_stride

        if self.first_stride is None:
            self.stride = (1, 1, 1)
        else:
            self.stride = self.first_stride

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *(
                [
                    ConvNormNonlin(
                        input_feature_channels,
                        output_feature_channels,
                        self.stride,
                        self.mode,
                        self.conv_type,
                        depth,
                    )
                ]
                + [
                    ConvNormNonlin(
                        output_feature_channels,
                        output_feature_channels,
                        (1, 1, 1),
                        self.mode,
                        self.conv_type,
                        None,
                    )
                    for _ in range(num_convs - 1)
                ]
            )
        )

    def forward(self, x):
        return self.blocks(x)


class ComplexUNet(nn.Module):
    def __init__(
        self, input_channels, base_num_features=32, num_layers=3, do_ds=False, mode='complex', conv_type='fast'
    ):
        super(ComplexUNet, self).__init__()

        self.conv_blocks_context = (
            []
        )  # this is used to get skip connections and store encoder convolutions
        self.conv_blocks_localization = []  # is used to store decoder convolutions
        self.tu = []  # is used to store upsamplings
        self.seg_outputs = []  # is used to store the outputs in each stage
        self.mode = mode
        self.conv_type = conv_type
        self.do_ds = do_ds  # for deep supervision

        if self.mode == "complex":
            transpconv = ComplexConvTranspose3d
            if self.conv_type == "fast":
                conv = ComplexConvFast3d
            else:
                conv = ComplexConv3d

        elif self.mode == "real":
            transpconv = nn.ConvTranspose3d

        input_features = input_channels
        output_features = base_num_features  # 32
        num_layers = num_layers

        # downsampling (encoder)
        for d in range(num_layers):
            if d != 0:  # determine the first stride
                first_stride = (2, 2, 2)
            else:
                first_stride = None
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_feature_channels=input_features,
                    output_feature_channels=output_features,
                    num_convs=2,
                    first_stride=first_stride,
                    mode=self.mode,
                    conv_type=self.conv_type,
                    depth=d,
                )
            )

            input_features = output_features
            output_features = int(np.round(output_features * 2))
            output_features = min(output_features, 320)  # 320 is max output feature

        # bottleneck
        first_stride = (2, 2, 2)
        final_num_features = output_features

        # Since we don't want to exceed the final num_feature, we use two blocks in the bottleneck.
        self.conv_blocks_context.append(
            nn.Sequential(
                StackedConvLayers(
                    input_feature_channels=input_features,
                    output_feature_channels=output_features,
                    num_convs=1,
                    first_stride=first_stride,
                    mode=self.mode,
                    conv_type=self.conv_type,
                    depth=None,
                ),
                StackedConvLayers(
                    input_feature_channels=output_features,
                    output_feature_channels=final_num_features,
                    num_convs=1,
                    first_stride=None,
                    mode=self.mode,
                    conv_type=self.conv_type,
                    depth=None,
                ),
            )
        )

        # upsampling
        for u in range(num_layers):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)
            ].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = (
                nfeatures_from_skip * 2
            )  # number of features after upsampling ang concatting skip connections

            final_num_features = nfeatures_from_skip

            self.tu.append(
                transpconv(
                    nfeatures_from_down, nfeatures_from_skip, (2, 2, 2), (2, 2, 2)
                )
            )
            self.conv_blocks_localization.append(
                nn.Sequential(
                    StackedConvLayers(
                        input_feature_channels=n_features_after_tu_and_concat,
                        output_feature_channels=nfeatures_from_skip,
                        num_convs=1,
                        first_stride=None,
                        mode=self.mode,
                        conv_type=self.conv_type,
                        depth=None,
                    ),
                    StackedConvLayers(
                        input_feature_channels=nfeatures_from_skip,
                        output_feature_channels=final_num_features,
                        num_convs=1,
                        first_stride=None,
                        mode=self.mode,
                        conv_type=self.conv_type,
                        depth=None,
                    ),
                )
            )

        if self.mode == "complex":
            for ds in range(len(self.conv_blocks_localization)):  # for deep supervision
                self.seg_outputs.append(
                    conv(
                        in_channels=self.conv_blocks_localization[ds][
                            -1
                        ].output_channels,
                        num_filters=3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

        elif self.mode == "real":
            for ds in range(len(self.conv_blocks_localization)):  # for deep supervision
                self.seg_outputs.append(
                    nn.Conv3d(
                        in_channels=self.conv_blocks_localization[ds][
                            -1
                        ].output_channels,
                        out_channels=3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def forward(self, x):
        skips = []
        seg_outputs = []

        # down
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        # bottleneck
        x = self.conv_blocks_context[-1](x)

        # up
        for u in range(len(self.tu)):
            x = self.tu[u](x)

            if self.mode == "complex":
                x = torch.cat((x, skips[-(u + 1)]), dim=2)
            elif self.mode == "real":
                x = torch.cat((x, skips[-(u + 1)]), dim=1)

            x = self.conv_blocks_localization[u](x)

            # output
            if self.mode == "complex":
                conv_for_softmax = self.seg_outputs[u](
                    x
                )  # applying last conv layer for softmax
                mag_out = (
                    conv_for_softmax[:, 0] ** 2 + conv_for_softmax[:, 1] ** 2
                )  # calculating the magnitude
                seg_outputs.append(mag_out)  # FaMo used softmax here. Since it was used with CrossEntropyLoss wich assumes logits, the softmax has been removed.

            elif self.mode == "real":
                seg_outputs.append(self.seg_outputs[u](x))

        if self.do_ds:
            return tuple(
                [seg_outputs[-1]]
                + [seg_outputs[i] for i in range(len(seg_outputs) - 2, -1, -1)]
            )
        else:
            return seg_outputs[-1]
