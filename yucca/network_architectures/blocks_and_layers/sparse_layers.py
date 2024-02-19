import torch
import torch.nn as nn


"""
This function takes the _cur_active variable, which is the 0/1 mask, and resizes (repeat_interleave) it
to the dimensions of the current input. E.g. it might be [2, 1, 64, 64] while the input is [2, 3, 128, 128].
it will then use repeat_interleave to turn this into a [2, 1, 128, 128] mask.

It can return either:
    ex - the "currently active" 0/1 mask, fitted to the shape of the current input.
        This is used in the sparse CONVOLUTIONS
    ii - the list of ACTIVE indices of the "currently active" 0/1 mask, fitted to the shape of the current input
        This is used in the sparse LAYER NORM
    """

_mask = None


def _get_mask_or_indices2d(mask, H, W, return_as_mask=True):
    h_repeat, w_repeat = H // mask.shape[-2], W // mask.shape[-1]
    mask = mask.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return mask if return_as_mask else mask.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def _get_mask_or_indices3d(mask, H, W, D, return_as_mask=True):
    h_repeat, w_repeat, d_repeat = (
        H // mask.shape[2],
        W // mask.shape[3],
        D // mask.shape[4],
    )
    mask = mask.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3).repeat_interleave(d_repeat, dim=4)
    return mask if return_as_mask else mask.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward3d(self, x: torch.Tensor):
    # (BCHWD) *= (B1HWD), mask the output of conv
    x = super(type(self), self).forward(x)
    x *= _get_mask_or_indices3d(_mask, H=x.shape[2], W=x.shape[3], D=x.shape[4], return_as_mask=True)
    return x


def sp_conv_forward2d(self, x: torch.Tensor):
    # (BCHW) *= (B1HW), mask the output of conv
    x = super(type(self), self).forward(x)
    x *= _get_mask_or_indices2d(_mask, H=x.shape[2], W=x.shape[3], return_as_mask=True)
    return x


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward2d


class SparseConv3d(nn.Conv3d):
    forward = sp_conv_forward3d


class SparseConvNeXtLayerNorm2d(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine=True)

    def forward(self, x):
        # channels_first, BCHW
        ii = _get_mask_or_indices2d(_mask, H=x.shape[2], W=x.shape[3], return_as_mask=False)
        bhwc = x.permute(0, 2, 3, 1)
        nc = bhwc[ii]
        nc = super(SparseConvNeXtLayerNorm2d, self).forward(nc)

        x = torch.zeros_like(bhwc)
        x[ii] = nc
        return x.permute(0, 3, 1, 2)


class SparseConvNeXtLayerNorm3d(nn.LayerNorm):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine=True)

    def forward(self, x):
        # channels_first, BCHW
        ii = _get_mask_or_indices3d(_mask, H=x.shape[2], W=x.shape[3], D=x.shape[4], return_as_mask=False)
        bhwc = x.permute(0, 2, 3, 4, 1)
        nc = bhwc[ii]
        nc = super(SparseConvNeXtLayerNorm3d, self).forward(nc)

        x = torch.zeros_like(bhwc)
        x[ii] = nc
        return x.permute(0, 4, 1, 2, 3)
