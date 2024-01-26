"""
UNETR based on: "Hatamizadeh et al.,
UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
"""

from typing import Tuple
import torch
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
from yucca.network_architectures.networks.YuccaNet import YuccaNet


class UNetR(YuccaNet):
    def __init__(
        self,
        input_channels: int,
        patch_size: list | Tuple,
        dropout_op_kwargs={"p": 0.0},
        num_classes: int = 1,
        starting_filters: int = 16,
        weightInitializer=None,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            num_classes: dimension of output channels.
            patch_size: dimension of input patch.
            starting_filters: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), starting_filters=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """
        super(UNetR, self).__init__()
        # Fixed parameters
        hidden_size = 768
        mlp_dim = 3072
        num_heads = 12
        pos_embed = "perceptron"
        norm_name = "instance"
        conv_block = False
        res_block = True

        if not (0 <= dropout_op_kwargs["p"] <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        # Task specific
        self.num_classes = num_classes
        self.filters = starting_filters

        # Model parameters
        self.weightInitializer = weightInitializer

        self.num_layers = 12
        if len(patch_size) == 2:
            self.transformer_patch_size = (16, 16)
            self.feat_size = (
                patch_size[0] // self.transformer_patch_size[0],
                patch_size[1] // self.transformer_patch_size[1],
            )
            self.spatial_dims = 2
        else:
            self.transformer_patch_size = (16, 16, 16)
            self.feat_size = (
                patch_size[0] // self.transformer_patch_size[0],
                patch_size[1] // self.transformer_patch_size[1],
                patch_size[2] // self.transformer_patch_size[2],
            )
            self.spatial_dims = 3
        self.hidden_size = hidden_size
        self.classification = False

        self.vit = ViT(
            in_channels=input_channels,
            img_size=patch_size,
            patch_size=self.transformer_patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_op_kwargs["p"],
            spatial_dims=self.spatial_dims,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=self.filters * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=self.filters * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=self.filters * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=self.filters * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters * 8,
            out_channels=self.filters * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters * 4,
            out_channels=self.filters * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=self.filters * 2,
            out_channels=self.filters,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=self.filters, out_channels=self.num_classes)  # type: ignore

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def proj_feat(self, x, hidden_size, feat_size):
        if len(feat_size) == 2:
            x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(weights["state_dict"]["module.transformer.patch_embedding.cls_token"])
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits
