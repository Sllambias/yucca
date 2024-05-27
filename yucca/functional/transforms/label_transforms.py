import numpy as np
import torch


def convert_labels_to_regions(data, regions):
    b, c, *shape = data.shape
    assert c == 1, "# Channels is not 1. Make sure the input to this function is a segmentation map of dims (b,c,h,w[,d])"
    region_canvas = np.zeros((b, len(regions), *shape))
    for channel, region in enumerate(regions):
        region_canvas[:, channel] = np.isin(data[:, 0], region)
    region_canvas = region_canvas.astype(np.uint8)
    return region_canvas


def convert_regions_to_labels(data, region_labels):
    b, _, *shape = data.shape
    region_canvas = np.zeros((b, 1, *shape))
    for channel, region_label in enumerate(region_labels):
        mask = data[:, channel] > 0.5
        region_canvas[:, 0][mask] = region_label
    region_canvas = region_canvas
    return region_canvas


def torch_convert_regions_to_labels(data, region_labels):
    b, _, *shape = data.shape
    region_canvas = torch.zeros((b, 1, *shape))
    for channel, region_label in enumerate(region_labels):
        mask = data[:, channel] > 0.5
        region_canvas[:, 0][mask] = region_label
    region_canvas = region_canvas
    return region_canvas
