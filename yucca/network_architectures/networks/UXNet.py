import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from yucca.network_architectures.networks.YuccaNet import YuccaNet
from yucca.network_architectures.blocks_and_layers.conv_blocks import UXNet_encoder
from yucca.network_architectures.blocks_and_layers.norm import (
    LayerNorm2d,
    LayerNorm3d,
)


class UXNet(YuccaNet):
    "https://github.com/MASILab/3DUX-Net/blob/main/networks/UXNet_3D/uxnet_encoder.py"

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        conv_op=nn.Conv2d,
        weightInitializer=None,
        deep_supervision=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            num_classes: dimension of output channels.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.


        """
        super(UXNet, self).__init__()

        if conv_op == nn.Conv2d:
            self.norm_op = LayerNorm2d
            self.spatial_dims = 2

        else:
            self.norm_op = LayerNorm3d
            self.spatial_dims = 3

        # Fixed parameters
        self.hidden_size = 768
        norm_name = "instance"
        res_block = True
        self.drop_path_rate = 0
        self.depths = [2, 2, 2, 2]
        self.feat_size = [48, 96, 192, 384]

        # Task specific
        self.num_classes = num_classes

        # Model parameters
        self.conv_op = conv_op
        self.norm_op_kwargs = {"eps": 1e-5}
        self.weightInitializer = weightInitializer
        self.deep_supervision = deep_supervision

        self.num_layers = 12

        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.uxnet = UXNet_encoder(
            in_channels=input_channels,
            depths=self.depths,
            dims=self.feat_size,
            conv_op=self.conv_op,
            norm_op=self.norm_op,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=input_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.ds_out_conv0 = self.conv_op(self.hidden_size, self.num_classes, kernel_size=1)

        self.decoder1 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.ds_out_conv1 = self.conv_op(self.feat_size[3], self.num_classes, kernel_size=1)

        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.ds_out_conv2 = self.conv_op(self.feat_size[2], self.num_classes, kernel_size=1)

        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.ds_out_conv3 = self.conv_op(self.feat_size[1], self.num_classes, kernel_size=1)

        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=self.feat_size[0], out_channels=self.num_classes)  # type: ignore

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x_in):
        outs = self.uxnet(x_in)

        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])

        dec1 = self.decoder1(enc_hidden, enc4)
        dec2 = self.decoder2(dec1, enc3)
        dec3 = self.decoder3(dec2, enc2)
        dec4 = self.decoder4(dec3, enc1)
        dec5 = self.decoder5(dec4)

        if self.deep_supervision:
            ds0 = self.ds_out_conv0(enc_hidden)
            ds1 = self.ds_out_conv1(dec1)
            ds2 = self.ds_out_conv2(dec2)
            ds3 = self.ds_out_conv3(dec3)
            ds4 = self.out(dec5)
            return [ds4, ds3, ds2, ds1, ds0]

        return self.out(dec5)
