import torch
import torch.nn as nn
from yucca.modules.optimization.loss_functions.nnUNet_losses import get_tp_fp_fn_tn


class SoftSigmoidDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=True, batch_dice=False, do_bg=True, smooth=1.0):
        """ """
        super(SoftSigmoidDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is True:
            x = torch.sigmoid(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc
