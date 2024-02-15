import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from yucca.network_architectures.blocks_and_layers.conv_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock,
)
from yucca.network_architectures.networks.YuccaNet import YuccaNet


class MedNeXt(YuccaNet):
    """
    From the paper: https://arxiv.org/pdf/2303.09975.pdf
    code source: https://github.com/MIC-DKFZ/MedNeXt/tree/main
    Using the 5-kernel L version presented in the paper.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        conv_op=None,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        checkpoint_style="outside_block",  # Either inside block or outside block
        block_counts: list = [
            3,
            4,
            8,
            8,
            8,
            8,
            8,
            4,
            3,
        ],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        assert checkpoint_style in [None, "outside_block"]
        if checkpoint_style == "outside_block":
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if conv_op == nn.Conv2d:
            dim = "2d"
        else:
            dim = "3d"

        self.stem = conv_op(input_channels, starting_filters, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * starting_filters,
            out_channels=16 * starting_filters,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 16,
                    out_channels=starting_filters * 16,
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.num_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        if self.deep_supervision:
            self.out_1 = OutBlock(in_channels=starting_filters * 2, n_classes=self.num_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=starting_filters * 4, n_classes=self.num_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=starting_filters * 8, n_classes=self.num_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=starting_filters * 16, n_classes=self.num_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.deep_supervision:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.deep_supervision:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.deep_supervision:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.deep_supervision:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.deep_supervision:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.deep_supervision:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.deep_supervision:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.deep_supervision:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.deep_supervision:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


class MedNeXt_Modular(YuccaNet):
    """
    From the paper: https://arxiv.org/pdf/2303.09975.pdf
    code source: https://github.com/MIC-DKFZ/MedNeXt/tree/main
    Using the 5-kernel L version presented in the paper.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        conv_op=None,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        checkpoint_style="outside_block",  # Either inside block or outside block
        block_counts: list = [
            3,
            4,
            8,
            8,
            8,
            8,
            8,
            4,
            3,
        ],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.encoder = MedNeXt_encoder(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_op=conv_op,
            starting_filters=starting_filters,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            block_counts=block_counts,
            norm_type=norm_type,
            grn=grn,
        )
        self.decoder = MedNeXt_decoder(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_op=conv_op,
            starting_filters=starting_filters,
            kernel_size=kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            checkpoint_style=checkpoint_style,
            block_counts=block_counts,
            norm_type=norm_type,
            grn=grn,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MedNeXt_encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        conv_op=None,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        checkpoint_style="outside_block",  # Either inside block or outside block
        block_counts: list = [
            3,
            4,
            8,
            8,
            8,
            8,
            8,
            4,
            3,
        ],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        assert checkpoint_style in [None, "outside_block"]
        if checkpoint_style == "outside_block":
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if conv_op == nn.Conv2d:
            dim = "2d"
        else:
            dim = "3d"

        self.stem = conv_op(input_channels, starting_filters, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[0],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[0])
            ]
        )

        self.down_0 = MedNeXtDownBlock(
            in_channels=starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
        )

        self.enc_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[1],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[1])
            ]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[2],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[2])
            ]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.enc_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[3],
                    kernel_size=enc_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[3])
            ]
        )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * starting_filters,
            out_channels=16 * starting_filters,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.bottleneck = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 16,
                    out_channels=starting_filters * 16,
                    exp_r=exp_r[4],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[4])
            ]
        )

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x):
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)
            x = self.iterative_checkpoint(self.bottleneck, x)
        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
        return [x_res_0, x_res_1, x_res_2, x_res_3, x]


class MedNeXt_decoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int = 1,
        conv_op=None,
        starting_filters: int = 32,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3],  # Expansion ratio as in Swin Transformers
        kernel_size: int = 5,  # Ofcourse can test kernel_size
        dec_kernel_size: int = None,
        deep_supervision: bool = False,  # Can be used to test deep supervision
        do_res: bool = True,  # Can be used to individually test residual connection
        do_res_up_down: bool = True,  # Additional 'res' connection on up and down convs
        checkpoint_style="outside_block",  # Either inside block or outside block
        block_counts: list = [
            3,
            4,
            8,
            8,
            8,
            8,
            8,
            4,
            3,
        ],  # Can be used to test staging ratio:
        norm_type="group",
        grn=False,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        assert checkpoint_style in [None, "outside_block"]
        if checkpoint_style == "outside_block":
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False
        if kernel_size is not None:
            dec_kernel_size = kernel_size

        if conv_op == nn.Conv2d:
            dim = "2d"
        else:
            dim = "3d"

        self.stem = conv_op(input_channels, starting_filters, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r for i in range(len(block_counts))]

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * starting_filters,
            out_channels=8 * starting_filters,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_3 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 8,
                    out_channels=starting_filters * 8,
                    exp_r=exp_r[5],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[5])
            ]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * starting_filters,
            out_channels=4 * starting_filters,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_2 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 4,
                    out_channels=starting_filters * 4,
                    exp_r=exp_r[6],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[6])
            ]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * starting_filters,
            out_channels=2 * starting_filters,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_1 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters * 2,
                    out_channels=starting_filters * 2,
                    exp_r=exp_r[7],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[7])
            ]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * starting_filters,
            out_channels=starting_filters,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )

        self.dec_block_0 = nn.Sequential(
            *[
                MedNeXtBlock(
                    in_channels=starting_filters,
                    out_channels=starting_filters,
                    exp_r=exp_r[8],
                    kernel_size=dec_kernel_size,
                    do_res=do_res,
                    norm_type=norm_type,
                    dim=dim,
                    grn=grn,
                )
                for i in range(block_counts[8])
            ]
        )

        self.out_0 = OutBlock(in_channels=starting_filters, n_classes=self.num_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        if self.deep_supervision:
            raise NotImplementedError

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, xs: list):
        # unpack the output of the encoder
        x_res_0, x_res_1, x_res_2, x_res_3, x = xs
        if self.outside_block_checkpointing:

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)

            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)

            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)

            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)

            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        return x
