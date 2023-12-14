import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.depthconv = conv_op(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.pointwiseconv = conv_op(input_channels, output_channels, kernel_size=1)


class ConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
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

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs["p"] is not None and self.dropout_op_kwargs["p"] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.activation = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(self.norm(x))


class ConvDropoutNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.norm(x)


class DoubleConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
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

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormNonlin(
            output_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
