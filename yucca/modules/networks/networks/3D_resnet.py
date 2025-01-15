from typing import Union, List, Optional, Callable, Type
from pytorchvideo.models.resnet import create_resnet
from torch import nn, Tensor
import torch
from yucca.modules.networks.blocks_and_layers.res_blocks import BasicBlock, conv_k1
from yucca.modules.networks.networks import YuccaNet


class ResNet(YuccaNet):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        in_channels,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        dropout_op: Optional[nn.Module] = None,
        dropout_kwargs: Optional[dict] = None,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    ) -> None:
        super().__init__()

        self.inplanes = 64
        self.dilation = 1

        if isinstance(conv_op, nn.Conv2d):
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv_op(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_op(self.inplanes)
        self.relu = nonlin(**nonlin_kwargs)
        self.layer1 = self._make_layer(
            block=block,
            blocks=layers[0],
            dropout_op=dropout_op,
            dropout_kwargs=dropout_kwargs,
            conv_op=conv_op,
            norm_op=norm_op,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            planes=64,
        )
        self.layer2 = self._make_layer(
            block=block,
            blocks=layers[1],
            conv_op=conv_op,
            dilate=replace_stride_with_dilation[0],
            dropout_op=dropout_op,
            dropout_kwargs=dropout_kwargs,
            norm_op=norm_op,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            planes=128,
            stride=2,
        )
        self.layer3 = self._make_layer(
            block=block,
            blocks=layers[2],
            conv_op=conv_op,
            dilate=replace_stride_with_dilation[1],
            dropout_op=dropout_op,
            dropout_kwargs=dropout_kwargs,
            norm_op=norm_op,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            planes=256,
            stride=2,
        )
        self.layer4 = self._make_layer(
            block=block,
            blocks=layers[3],
            conv_op=conv_op,
            dilate=replace_stride_with_dilation[2],
            dropout_op=dropout_op,
            dropout_kwargs=dropout_kwargs,
            norm_op=norm_op,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            planes=512,
            stride=2,
        )
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) or isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        blocks: int,
        conv_op,
        dropout_op,
        dropout_kwargs,
        norm_op,
        nonlin,
        nonlin_kwargs,
        planes: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_k1(conv_op, self.inplanes, planes * block.expansion, stride),
                norm_op(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                base_width=self.base_width,
                conv_op=conv_op,
                dilation=previous_dilation,
                downsample=downsample,
                dropout_op=dropout_op,
                dropout_kwargs=dropout_kwargs,
                groups=self.groups,
                inplanes=self.inplanes,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                norm_op=norm_op,
                planes=planes,
                stride=stride,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    base_width=self.base_width,
                    conv_op=conv_op,
                    dilation=self.dilation,
                    dropout_op=dropout_op,
                    dropout_kwargs=dropout_kwargs,
                    groups=self.groups,
                    inplanes=self.inplanes,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    norm_op=norm_op,
                    planes=planes,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet50_3D(input_channels: int, num_classes: int):
    return create_resnet(input_channel=input_channels, model_depth=50, model_num_class=num_classes)


def ResNet50_Volumetric(input_channels: int, num_classes: int):
    return create_resnet(
        input_channel=input_channels,
        model_depth=50,
        model_num_class=num_classes,
        stem_conv_kernel_size=(7, 7, 7),
        stem_conv_stride=(2, 2, 2),
        stem_pool_kernel_size=(3, 3, 3),
        stem_pool_stride=(2, 2, 2),
        stage1_pool_kernel_size=(1, 1, 1),
        stage_conv_a_kernel_size=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        stage_conv_b_kernel_size=(
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        stage_spatial_h_stride=(1, 1, 1, 1),
        stage_spatial_w_stride=(1, 1, 1, 1),
        head_pool_kernel_size=(7, 7, 7),
    )


def ResNet50_Volumetric(input_channels: int, num_classes: int):
    model = create_resnet(
        input_channel=input_channels,
        model_depth=50,
        model_num_class=num_classes,
        stem_conv_kernel_size=(7, 7, 7),
        stem_conv_stride=(2, 2, 2),
        stem_pool_kernel_size=(3, 3, 3),
        stem_pool_stride=(2, 2, 2),
        stage1_pool_kernel_size=(1, 1, 1),
        stage_conv_a_kernel_size=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        stage_conv_b_kernel_size=(
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
            (3, 3, 3),
        ),
        stage_spatial_h_stride=(1, 1, 1, 1),
        stage_spatial_w_stride=(1, 1, 1, 1),
        head_pool_kernel_size=(7, 7, 7),
    )
    model.predict = model.forward
    return model


def resnet18(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    dropout_op=None,
    dropout_kwargs={"p": 0.25},
    norm_op=nn.InstanceNorm3d,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """

    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=input_channels,
        num_classes=num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        conv_op=conv_op,
        dropout_op=dropout_op,
        dropout_kwargs=dropout_kwargs,
        norm_op=norm_op,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
    )


def resnet18_dropout(
    input_channels: int,
    num_classes: int = 1,
    conv_op=nn.Conv3d,
    dropout_op=nn.Dropout3d,
    dropout_kwargs={"p": 0.25},
    norm_op=nn.InstanceNorm3d,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=input_channels,
        num_classes=num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        conv_op=conv_op,
        dropout_op=dropout_op,
        dropout_kwargs=dropout_kwargs,
        norm_op=norm_op,
        nonlin=nonlin,
        nonlin_kwargs=nonlin_kwargs,
    )


if __name__ == "__main__":
    net = resnet18(
        2,
        1,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        nonlin=nn.LeakyReLU,
        replace_stride_with_dilation=[False, False, False],
    )
    data = torch.zeros((2, 1, 64, 64, 64))
    out = net(data)
