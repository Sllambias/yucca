import torch
import torch.nn as nn
from yucca.network_architectures.networks.YuccaNet import YuccaNet
from yucca.network_architectures.blocks_and_layers.conv_layers import (
    DoubleConvDropoutNormNonlin,
)


class UNet(YuccaNet):
    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        starting_filters: int = 64,
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
        super(UNet, self).__init__()

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

        # For parameter groups
        self.max_block_number = 9  # amount of calls to self.basic_block(...)
        self.num_encoder_blocks = 4  # number of self.encoder_convX blocks

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

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            self.filters * 4,
            self.filters * 8,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            self.filters * 8,
            self.filters * 16,
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

        self.upsample1 = self.upsample(self.filters * 16, self.filters * 8, kernel_size=2, stride=2)
        self.decoder_conv1 = self.basic_block(
            self.filters * 16,
            self.filters * 8,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(self.filters * 8, self.filters * 4, kernel_size=2, stride=2)
        self.decoder_conv2 = self.basic_block(
            self.filters * 8,
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

        self.upsample3 = self.upsample(self.filters * 4, self.filters * 2, kernel_size=2, stride=2)
        self.decoder_conv3 = self.basic_block(
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

        self.upsample4 = self.upsample(self.filters * 2, self.filters, kernel_size=2, stride=2)
        self.decoder_conv4 = self.basic_block(
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

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(self.filters * 16, self.num_classes, kernel_size=1)
            self.ds_out_conv1 = self.conv_op(self.filters * 8, self.num_classes, kernel_size=1)
            self.ds_out_conv2 = self.conv_op(self.filters * 4, self.num_classes, kernel_size=1)
            self.ds_out_conv3 = self.conv_op(self.filters * 2, self.num_classes, kernel_size=1)

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

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        # Decoder path
        x5 = torch.cat([self.upsample1(x4), x3], dim=1)
        x5 = self.decoder_conv1(x5)

        x6 = torch.cat([self.upsample2(x5), x2], dim=1)
        x6 = self.decoder_conv2(x6)

        x7 = torch.cat([self.upsample3(x6), x1], dim=1)
        x7 = self.decoder_conv3(x7)

        x8 = torch.cat([self.upsample4(x7), x0], dim=1)
        x8 = self.decoder_conv4(x8)

        if self.deep_supervision:
            ds0 = self.ds_out_conv0(x4)
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.out_conv(x8)
            return [ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x8)
        return logits

    def get_parameter_groups(self, base_lr, lr_scale, print_parameter_groups=False):
        assert not self.deep_supervision, "Parameter groups for deep supervision is not yet supported... Sorry :-)"

        def param_group(name, k, lr, lr_scale):
            return {"name": name, "params": [], "param_names": [], "k": k, "lr_scale": lr_scale, "lr": lr * lr_scale}

        param_groups = {}

        for param_name, params in dict(self.named_parameters()).items():
            name_parts = param_name.split(".")
            section = name_parts[0]

            section_number = int(section[-1]) if section[-1] != "v" else None

            # Add param group without params if not present
            if section.startswith("upsample"):
                # upsamplex is part of decocder_convx group
                section_name = "decoder_conv" + str(section_number)
            elif section.startswith("ds") and not self.deep_supervision:
                assert not self.deep_supervision
                continue
            else:
                section_name = section

            if section_name not in param_groups.keys():
                if section_name == "in_conv":
                    k = self.max_block_number
                elif section_name == "out_conv":
                    k = 0
                elif "encoder" in section_name:
                    k = self.max_block_number - section_number
                elif "decoder" in section_name:
                    k = self.num_encoder_blocks - section_number
                else:
                    raise NotImplementedError("Parameters are not yet supported using parameter groups:", param_name)

                param_groups[section_name] = param_group(section_name, k, base_lr, lr_scale**k)

            # Add parameters to param group
            param_groups[section_name]["params"].append(params)
            param_groups[section_name]["param_names"].append(param_name)

        parameters = list(param_groups.values())

        if print_parameter_groups:
            params_sorted = sorted(parameters, key=lambda x: x["k"])
            indent = " " * 4

            for p in params_sorted:
                print(p["name"])
                print(indent, "k:", p["k"])
                print(indent, "lr_scale:", p["lr_scale"])
                print(indent, "lr:", p["lr"])
                print(indent, "param_name:", p["param_names"])

        return parameters
