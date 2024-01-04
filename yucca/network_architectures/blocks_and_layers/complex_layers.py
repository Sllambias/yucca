from yucca.network_architectures.utils.complex_values import (
    Cmul
)


##############################################################################################
#                                                                                            #
#     coded by FaMo (faezeh.mosayyebi@gmail.com)                                             #
#     Description: The Following classes are the essential layers to make a complex valued   #
#     U-net. To get more information please refer to the documentation.                      #
#                                                                                            #
##############################################################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ComplexConv3d(nn.Module):
    # Our complex convolution implementation
    def __init__(
        self,
        in_channels,
        num_filters,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ComplexConv3d, self).__init__()

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.A = nn.Conv3d(
            self.in_channels,
            self.num_filters,
            self.kernel_size,
            self.stride,
            self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.B = nn.Conv3d(
            self.in_channels,
            self.num_filters,
            self.kernel_size,
            self.stride,
            self.padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Initialize wights
        fan_in = True
        if fan_in:
            c = self.in_channels
        else:
            c = self.num_filters

        gain = 1 / np.sqrt(2)
        with torch.no_grad():
            std = gain / np.sqrt(self.kernel_size * self.kernel_size * c)
            self.A.weight.normal_(0, std)
            self.B.weight.normal_(0, std)

    def __repr__(self):
        return "ComplexConv3D"

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        if len(x.shape) == 6:
            real = x[:, 0]
            imag = x[:, 1]
            out_real = self.A(real) - self.B(imag)
            out_imag = self.B(real) + self.A(imag)
            return torch.stack([out_real, out_imag], dim=1)
        else:
            out_real = self.A(x)
            out_imag = self.B(x)
            return torch.stack([out_real, out_imag], dim=1)


class ComplexConvFast3d(nn.Module):
    # Our complex convolution implementation
    def __init__(
        self,
        in_channels,
        num_filters,
        kernel_size,
        stride,
        padding,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ComplexConvFast3d, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.A = nn.Conv3d(
            self.in_channels,
            self.num_filters,
            self.kernel_size,
            self.stride,
            self.padding,
            groups=groups,
            bias=bias,
        )
        self.B = nn.Conv3d(
            self.in_channels,
            self.num_filters,
            self.kernel_size,
            self.stride,
            self.padding,
            groups=groups,
            bias=bias,
        )

        # Initialize wights
        fan_in = True
        if fan_in:
            c = self.in_channels
        else:
            c = self.num_filters

        gain = 1 / np.sqrt(2)
        with torch.no_grad():
            std = gain / np.sqrt(self.kernel_size * self.kernel_size * c)
            self.A.weight.normal_(0, std)
            self.B.weight.normal_(0, std)

    def __repr__(self):
        return "ComplexConv3d"

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        if len(x.shape) == 6:
            real = x[:, 0]
            imag = x[:, 1]
            t1 = self.A(real)
            t2 = self.B(imag)

            t3 = F.conv3d(
                real + imag,
                weight=(self.A.weight + self.B.weight),
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )

            return torch.stack([t1 - t2, t3 - t1 - t2], dim=1)
        else:
            out_real = self.A(x)
            out_imag = self.B(x)
            return torch.stack([out_real, out_imag], dim=1)


class ComplexConvTranspose3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(ComplexConvTranspose3d, self).__init__()

        # Convolution parameters
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        ## Model components
        self.A = nn.ConvTranspose3d(
            self.in_channel,
            self.out_channel,
            self.kernel_size,
            self.stride,
            padding=0,
            output_padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )
        self.B = nn.ConvTranspose3d(
            self.in_channel,
            self.out_channel,
            self.kernel_size,
            self.stride,
            padding=0,
            output_padding=0,
            groups=1,
            bias=False,
            dilation=1,
        )

    def __repr__(self):
        return "ComplexConvTranspose3d"

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        if len(x.shape) == 6:
            real = x[:, 0]
            imag = x[:, 1]
            out_real = self.A(real) - self.B(imag)
            out_imag = self.B(real) + self.A(imag)
            return torch.stack([out_real, out_imag], dim=1)
        else:
            out_real = self.A(x)
            out_imag = self.B(x)
            return torch.stack([out_real, out_imag], dim=1)


class ComplexInstanceNorm3d(nn.Module):
    """
    Equivariant Complex Instance Norm
    Computes magnitude of the complex input and applies Instance norm on it
    """

    def __init__(self, channels, eps, momentum, affine):
        super(ComplexInstanceNorm3d, self).__init__()
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.IN = nn.InstanceNorm3d(channels, self.eps, self.momentum, self.affine)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        mag = torch.norm(x, dim=1)
        normalized = self.IN(mag)
        mag_factor = normalized / (mag + 1e-6)
        return x * mag_factor[:, None, ...]


class ComplexBatchNorm3D(nn.Module):
    """
    Equivariant Complex Batch Norm
    Computes magnitude of the complex input and applies batch norm on it
    """

    def __init__(self, channels, eps, momentum, affine):
        super(ComplexBatchNorm3D, self).__init__()

        self.bn = nn.BatchNorm3d(channels, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        mag = torch.norm(x, dim=1)
        normalized = self.bn(mag)
        mag_factor = normalized / (mag + 1e-6)
        return x * mag_factor[:, None, ...]


class NaiveCBN(nn.Module):
    """
    Naive BatchNorm which concatenates real and imaginary channels
    """

    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=True):
        super(NaiveCBN, self).__init__()
        self.bn = nn.BatchNorm3d(
            channels * 2, eps=eps, momentum=momentum, affine=affine
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        x_shape = x.shape
        return self.bn(
            x.reshape(
                x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )
        ).reshape(x_shape)


class scaling_layer(nn.Module):
    """
    scaling layer for GTReLU
    """

    def __init__(self, channels):
        super(scaling_layer, self).__init__()
        self.a_bias = nn.Parameter(
            torch.rand(
                channels,
            ),
            requires_grad=True,
        )
        self.b_bias = nn.Parameter(
            torch.rand(
                channels,
            ),
            requires_grad=True,
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        x_c = x[:, 0]
        x_d = x[:, 1]

        a_bias = self.a_bias[None, :, None, None, None]
        b_bias = self.b_bias[None, :, None, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        return torch.stack([real_component, imag_component], dim=1)


class Two_Channel_Nonlinearity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        temp_phase = inputs
        phase_mask = temp_phase % (
            2 * np.pi
        )  # This ensures that the values are wrapped within the range [0, 2*np.pi)
        phase_mask = (phase_mask <= np.pi).type(torch.cuda.FloatTensor) * (
            phase_mask >= 0
        ).type(
            torch.cuda.FloatTensor
        )  # make a mask to set the phase out of [0,pi] 0
        temp_phase = temp_phase * phase_mask

        ctx.save_for_backward(inputs, phase_mask)

        return temp_phase

    @staticmethod
    def backward(ctx, grad_output):
        inputs, phase_mask = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input * (1 - phase_mask)

        return grad_input


class GTReLU(nn.Module):
    """
    GTReLU layer
    """

    def __init__(self, channels, negative_slope=1e-2, inplace=True):
        super(GTReLU, self).__init__()

        # c = (a_bias) + (b_bias)j
        self.a_bias = nn.Parameter(
            torch.rand(
                channels,
            ),
            requires_grad=True,
        )  # size = channels
        self.b_bias = nn.Parameter(
            torch.rand(
                channels,
            ),
            requires_grad=True,
        )

        self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        self.cn = Two_Channel_Nonlinearity.apply
        self.phase_scale = nn.Parameter(
            torch.ones(
                channels,
            )
        )

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        x_c = x[:, 0]
        x_d = x[:, 1]

        # Scaling (c)
        a_bias = self.a_bias[
            None, :, None, None, None
        ]  # size (1,channels,1,1,1) = (b, c, x, y, z)
        b_bias = self.b_bias[None, :, None, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        x = torch.stack([real_component, imag_component], dim=1)

        # Thresholding
        temp_abs = torch.norm(x, dim=1)
        temp_phase = torch.atan2(
            x[:, 1, ...], x[:, 0, ...] + (x[:, 0, ...] == 0) * 1e-5
        )

        final_abs = temp_abs.unsqueeze(1)

        final_phase = self.cn(temp_phase)

        x = torch.cat(
            (
                final_abs * torch.cos(final_phase).unsqueeze(1),
                final_abs * torch.sin(final_phase).unsqueeze(1),
            ),
            1,
        )

        # Phase scaling [Optional]
        norm = torch.norm(x, dim=1)
        angle = torch.atan2(x[:, 1], x[:, 0] + (x[:, 0] == 0) * 1e-5)
        angle = angle * torch.minimum(
            torch.maximum(
                self.phase_scale[None, :, None, None, None], torch.tensor(0.5)
            ),
            torch.tensor(2.0),
        )

        x = torch.stack([norm * torch.cos(angle), norm * torch.sin(angle)], dim=1)

        return x


class eqnl(nn.Module):
    def __init__(self, channels, negative_slope=1e-2, inplace=True):
        # Applies tangent reLU to inputs.
        super(eqnl, self).__init__()
        self.phase_scale = nn.Parameter(
            torch.ones(
                channels,
            ),
            requires_grad=True,
        )  # Phase scale (w)
        self.cn = Two_Channel_Nonlinearity.apply  # To apply phase treshold
        self.lrelu = nn.LeakyReLU(negative_slope, inplace)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """
        p1 = x  # f in figure 3
        p2 = torch.mean(x, dim=2, keepdim=True)  # m in figure 3

        abs1 = torch.norm(p1, dim=1, keepdim=True) + 1e-6  # |f|
        abs2 = torch.norm(p2, dim=1, keepdim=True) + 1e-6  # |m|
        p2 = p2 / abs2  # m hat

        conjp2 = torch.stack((p2[:, 0], -p2[:, 1]), 1)  # (m hat)*
        shifted = Cmul(p1, conjp2)  # f * (m hat)*
        phasediff = torch.atan2(
            shifted[:, 1], shifted[:, 0] + (shifted[:, 0] == 0) * 1e-5
        )  # calculating the phase of f * (m hat)*
        final_phase = self.cn(phasediff) * self.lrelu(
            self.phase_scale[None, :, None, None, None].clone()
        )  # thersholding and scaling by a trainable scaler | th.relu

        out = abs1 * Cmul(
            torch.stack([torch.cos(final_phase), torch.sin(final_phase)], 1), p2
        )  # multiplying by unchanged |f| to get fout

        return out


class DivLayer(nn.Module):
    """
    division layer
    """

    def __init__(
        self,
        output_channels,
        kern_size,
        conv_type,
        stride=(1, 1, 1),
        padding=1,
        dilation=1,
        groups=1,
    ):
        super(DivLayer, self).__init__()

        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_one_filter = False

        if conv_type == "fast":
            usedconv = ComplexConvFast3d
        else:
            usedconv = ComplexConv3d

        if self.use_one_filter:
            self.conv = usedconv(
                output_channels, 1, kern_size, stride, padding, dilation, groups
            )
        else:
            self.conv = usedconv(
                output_channels,
                output_channels,
                kern_size,
                stride,
                padding,
                dilation,
                groups,
            )

    def __repr__(self):
        return "DivLayer"

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """

        if self.use_one_filter:
            conv = self.conv(x)
            conv = conv.repeat(
                1, 1, x.shape[2], 1, 1, 1
            )  # reapet the tensor over each channel
        else:
            convresult = self.conv(x)

        a, b = x[:, 0], x[:, 1]
        c, d = convresult[:, 0], convresult[:, 1]

        divisor = c**2 + d**2 + 1e-7

        real = (a * c + b * d) / divisor  # ac + bd
        imag = (b * c - a * d) / divisor  # (bc - ad)i

        return torch.stack([real, imag], dim=1)


class ConjugateLayer(nn.Module):
    """
    conjugate layer
    """

    def __init__(
        self,
        output_channels,
        kern_size,
        conv_type,
        stride=(1, 1, 1),
        padding=1,
        dilation=1,
        groups=1,
    ):
        super(ConjugateLayer, self).__init__()
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_one_filter = False

        if conv_type == "fast":
            usedconv = ComplexConvFast3d
        else:
            usedconv = ComplexConv3d

        if self.use_one_filter:
            self.conv = usedconv(
                output_channels, 1, kern_size, stride, padding, dilation, groups
            )
        else:
            self.conv = usedconv(
                output_channels,
                output_channels,
                kern_size,
                stride,
                padding,
                dilation,
                groups,
            )

    def __repr__(self):
        return "Conjugate"

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, X, Y, Z]
        """

        if self.use_one_filter:
            conv = self.conv(x)
            conv = conv.repeat(1, 1, x.shape[2], 1, 1, 1)
        else:
            convresult = self.conv(x)

        a, b = x[:, 0], x[:, 1]
        c, d = convresult[:, 0], convresult[:, 1]

        real = a * c + b * d  # ac + bd
        imag = b * c - a * d  # (bc - ad)i

        return torch.stack([real, imag], dim=1)
