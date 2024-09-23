import torch
from torch import nn
from yucca.modules.optimization.loss_functions.CE import CE
from yucca.modules.optimization.loss_functions.nnUNet_losses import SoftDiceLoss, sum_tensor
import torch.nn.functional as F


class SoftSkeletonRecallLoss(nn.Module):
    def __init__(self, apply_softmax=True, batch_dice=False, do_bg=True, smooth=1.0):
        super(SoftSkeletonRecallLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_softmax = apply_softmax
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_softmax is True:
            x = F.softmax(x)

        x = x[:, 1:]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=y.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

            sum_gt = sum_tensor(y_onehot, axes) if loss_mask is None else sum_tensor(y_onehot * loss_mask, axes)

        inter_rec = sum_tensor((x * y_onehot), axes) if loss_mask is None else sum_tensor(x * y_onehot * loss_mask, axes)

        rec = (inter_rec + self.smooth) / (sum_gt + self.smooth)

        rec = rec.mean()
        return -rec


class DC_SkelREC_and_CE_loss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs={},
        soft_skelrec_kwargs={},
        ce_kwargs={},
        weight_ce=1,
        weight_dice=1,
        weight_srec=1,
        log_dice=False,
        ignore_label=None,
    ):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param soft_skelrec_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_SkelREC_and_CE_loss, self).__init__()

        if ignore_label is not None:
            ce_kwargs["reduction"] = "none"
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        self.ce = CE()
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(**soft_skelrec_kwargs)

    def forward(self, net_output, target, skel):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "not implemented for one hot encoding"
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0

        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        srec_loss = self.srec(net_output, skel) if self.weight_srec != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_srec * srec_loss
        return result
