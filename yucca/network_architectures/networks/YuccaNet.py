import torch
from torch import nn
from yucca.utils.torch_utils import get_available_device
from yucca.network_architectures.utils.get_steps_for_sliding_window import (
    get_steps_for_sliding_window,
)


class YuccaNet(nn.Module):
    def __init__(self):
        super(YuccaNet, self).__init__()
        # Attributes that need to be defined for each architecture
        self.abbreviation: str = None
        self.dimensions: int = None

    def forward(self):
        """
        implement in individual trainers.
        DO NOT INCLUDE FINAL SOFTMAX/SIGMOID ETC.
        WILL BE HANDLED BY LOSS FUNCTIONS
        """
        pass

    def load_state_dict(self, target_state_dict, *args, **kwargs):
        current_state_dict = self.state_dict()
        # filter unnecessary keys
        target_state_dict = {
            k: v
            for k, v in target_state_dict.items()
            if (k in current_state_dict) and (current_state_dict[k].shape == target_state_dict[k].shape)
        }
        super().load_state_dict(target_state_dict, *args, **kwargs)

    def predict(self, mode, data, patch_size, overlap, sliding_window_prediction=True, mirror=False):
        data = data.to(torch.device(get_available_device()))

        if not sliding_window_prediction:
            return self._full_image_predict(data)

        if mode == "3D":
            predict = self._sliding_window_predict3D
        elif mode == "2D":
            predict = self._sliding_window_predict2D

        pred = predict(data, patch_size, overlap)
        if mirror:
            pred += torch.flip(predict(torch.flip(data, (2,)), patch_size, overlap), (2,))
            pred += torch.flip(predict(torch.flip(data, (3,)), patch_size, overlap), (3,))
            pred += torch.flip(predict(torch.flip(data, (2, 3)), patch_size, overlap), (2, 3))
            div = 4
            if mode == "3D":
                pred += torch.flip(predict(torch.flip(data, (4,)), patch_size, overlap), (4,))
                pred += torch.flip(predict(torch.flip(data, (2, 4)), patch_size, overlap), (2, 4))
                pred += torch.flip(predict(torch.flip(data, (3, 4)), patch_size, overlap), (3, 4))
                pred += torch.flip(
                    predict(torch.flip(data, (2, 3, 4)), patch_size, overlap),
                    (2, 3, 4),
                )
                div += 4
            pred /= div
        return pred

    def _full_image_predict(self, data):
        """
        Standard prediction used in cases where models predict on full-size images.
        This is opposed to patch-based predictions where we use a sliding window approach to generate
        full size predictions.
        """
        return self.forward(data)

    def _sliding_window_predict3D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=torch.device(get_available_device()),
        )

        x_steps, y_steps, z_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)
        px, py, pz = patch_size

        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    # check if out of bounds
                    out = self.forward(data[:, :, xs : xs + px, ys : ys + py, zs : zs + pz])
                    canvas[:, :, xs : xs + px, ys : ys + py, zs : zs + pz] += out
        return canvas

    def _sliding_window_predict2D(self, data, patch_size, overlap):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=torch.device(get_available_device()),
        )

        px, py = patch_size

        # If we have 5 dimensions we are working with 3D data, and need to predict each slice.
        if len(data.shape) == 5:
            x_steps, y_steps = get_steps_for_sliding_window(data.shape[3:], patch_size, overlap)
            for idx in range(data.shape[2]):
                for xs in x_steps:
                    for ys in y_steps:
                        out = self.forward(data[:, :, idx, xs : xs + px, ys : ys + py])
                        canvas[:, :, idx, xs : xs + px, ys : ys + py] += out
            return canvas

        # else we proceed with the data as 2D
        x_steps, y_steps = get_steps_for_sliding_window(data.shape[2:], patch_size, overlap)

        for xs in x_steps:
            for ys in y_steps:
                # check if out of bounds
                out = self.forward(data[:, :, xs : xs + px, ys : ys + py])
                canvas[:, :, xs : xs + px, ys : ys + py] += out
        return canvas


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if (
            isinstance(module, nn.Conv3d)
            or isinstance(module, nn.Conv2d)
            or isinstance(module, nn.ConvTranspose2d)
            or isinstance(module, nn.ConvTranspose3d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
